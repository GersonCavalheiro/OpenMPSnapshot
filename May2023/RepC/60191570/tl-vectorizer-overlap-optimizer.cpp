#include "tl-vectorizer-overlap-optimizer.hpp"
#include "tl-vectorizer-visitor-expression.hpp"
#include "tl-vectorization-utils.hpp"
#include "tl-vectorization-analysis-interface.hpp"
#include "tl-nodecl-utils.hpp"
#include "tl-optimizations.hpp"
#include "tl-expression-reduction.hpp"
#include "hlt-loop-unroll.hpp"
#include "cxx-cexpr.h"
namespace TL
{
namespace Vectorization
{
VectorizationAnalysisInterface *OverlappedAccessesOptimizer::_analysis = 0;
OverlappedAccessesOptimizer::OverlappedAccessesOptimizer(
VectorizerEnvironment& environment,
VectorizationAnalysisInterface *analysis,
const bool is_omp_simd_for,
const bool is_epilog,
const bool overlap_in_place,
Nodecl::List& prependix_stmts)
: _environment(environment), _is_omp_simd_for(is_omp_simd_for),
_is_simd_epilog(is_epilog), _in_place(overlap_in_place),
_prependix_stmts(prependix_stmts), _first_analysis(analysis)
{
_analysis = analysis;
}
void OverlappedAccessesOptimizer::update_alignment_info(
const Nodecl::NodeclBase& main_loop,
const Nodecl::NodeclBase& if_epilog)
{
Nodecl::NodeclBase func_code = 
Nodecl::Utils::get_enclosing_function(
main_loop).get_function_code();
Optimizations::canonicalize_and_fold(func_code, false );
_analysis = new VectorizationAnalysisInterface(
func_code,
Analysis::WhichAnalysis::INDUCTION_VARS_ANALYSIS);
objlist_nodecl_t main_vector_loads = Nodecl::Utils::
nodecl_get_all_nodecls_of_kind<Nodecl::VectorLoad>(main_loop);
objlist_nodecl_t epilog_vector_loads = Nodecl::Utils::
nodecl_get_all_nodecls_of_kind<Nodecl::VectorLoad>(if_epilog);
objlist_nodecl_t::iterator main_it = main_vector_loads.begin();
for(objlist_nodecl_t::iterator epilog_it = epilog_vector_loads.begin();
epilog_it != epilog_vector_loads.end();
main_it++, epilog_it++)
{
int alignment_output;
Nodecl::VectorLoad main_vl = main_it->as<Nodecl::VectorLoad>();
Nodecl::VectorLoad epilog_vl = epilog_it->as<Nodecl::VectorLoad>();
Nodecl::List flags = main_vl.get_flags().as<Nodecl::List>();
if(_analysis->is_simd_aligned_access(
_environment._analysis_simd_scope,
Utils::get_scalar_memory_access(main_vl),
_environment._aligned_symbols_map,
_environment._suitable_exprs_list,
1, 
main_vl.get_type().get_size(),
alignment_output) &&
flags.find_first<Nodecl::AlignedFlag>().is_null())
{
flags.append(Nodecl::AlignedFlag::make());
VECTORIZATION_DEBUG()
{
fprintf(stderr, "%s (aligned)\n", main_vl.prettyprint().c_str());
}
}
else if (alignment_output != -1 &&
flags.find_first<Nodecl::AlignmentInfo>().is_null())
{
flags.append(Nodecl::AlignmentInfo::make(
const_value_get_signed_int(alignment_output)));
fprintf(stderr, "%s (alignment info = %d)\n",
main_vl.prettyprint().c_str(), alignment_output);
}
main_vl.set_flags(flags);
epilog_vl.set_flags(flags.shallow_copy());
}
while(main_it != main_vector_loads.end())
{
int alignment_output;
Nodecl::VectorLoad main_vl = main_it->as<Nodecl::VectorLoad>();
Nodecl::List flags = main_vl.get_flags().as<Nodecl::List>();
if(_analysis->is_simd_aligned_access(
_environment._analysis_simd_scope,
Utils::get_scalar_memory_access(main_vl),
_environment._aligned_symbols_map,
_environment._suitable_exprs_list,
1, 
main_vl.get_type().get_size(),
alignment_output) &&
flags.find_first<Nodecl::AlignedFlag>().is_null())
{
flags.append(Nodecl::AlignedFlag::make());
VECTORIZATION_DEBUG()
{
fprintf(stderr, "%s (aligned)\n", main_vl.prettyprint().c_str());
}
}
else if (alignment_output != -1 &&
flags.find_first<Nodecl::AlignmentInfo>().is_null())
{
flags.append(Nodecl::AlignmentInfo::make(
const_value_get_signed_int(alignment_output)));
fprintf(stderr, "%s (alignment info = %d)\n",
main_vl.prettyprint().c_str(), alignment_output);
}
main_vl.set_flags(flags);
main_it++;
}
}
bool OverlappedAccessesOptimizer::need_init_cache(
const bool is_nested_loop,
const bool is_simd_epilog,
const bool is_overlap_epilog)
{
if (_is_omp_simd_for)
{
if (!is_nested_loop) 
{
return true; 
}
else 
{
if (is_overlap_epilog)
return false;   
else
return true;    
}
}
else 
{
if (!is_nested_loop) 
{
if (is_simd_epilog)
return false; 
else
return true; 
}
else 
{
if (is_overlap_epilog)
return false; 
else
return true;  
}
}
fatal_error("Overlap: Init cache missing case\n");
}
bool OverlappedAccessesOptimizer::need_update_post(
const bool is_nested_loop,
const bool is_simd_epilog,
const bool is_overlap_epilog)
{
if (!is_nested_loop) 
{
if (is_simd_epilog)
return false;
else
return true;
}
else 
{
if (is_overlap_epilog)
return false;
else
return true;
}
fatal_error("Overlap: Cache update post missing case\n");
}
Nodecl::List OverlappedAccessesOptimizer::get_ogroup_init_statements(
const OverlapGroup& ogroup,
const Nodecl::ForStatement& for_stmt,
const bool is_simd_loop, 
const bool is_omp_simd_for) const 
{
const objlist_nodecl_t& ivs_list = OverlappedAccessesOptimizer::
_analysis->get_linear_nodecls(for_stmt);
TL::Scope scope = for_stmt.retrieve_context();
Nodecl::List result_list;
Nodecl::List prefetching_list;
int num_init_registers;
bool gen_init_prefetching;
if (ogroup._inter_it_overlap == 1)
{
num_init_registers = ogroup._num_registers -1;
gen_init_prefetching = false; 
}
else
{  
num_init_registers = ogroup._num_registers;
gen_init_prefetching = false;
}
for (int i = 0; i < num_init_registers; i++)
{
Nodecl::NodeclBase vload_index =
ogroup._registers_indexes[i].shallow_copy();
if (ogroup._inter_it_overlap)
{
for (objlist_nodecl_t::const_iterator iv = ivs_list.begin();
iv != ivs_list.end();
iv++)
{
Nodecl::NodeclBase iv_lb;
if (is_simd_loop && is_omp_simd_for)
{
iv_lb = *iv;
}
else
{
iv_lb = OverlappedAccessesOptimizer::_analysis->
get_induction_variable_lower_bound(
for_stmt,*iv);
}
if (!iv_lb.is_null())
{
Nodecl::Utils::nodecl_replace_nodecl_by_structure(
vload_index, *iv, iv_lb);
}
}
}
Nodecl::List flags;
if (ogroup._aligned_strategy)
flags = Nodecl::List::make(
Nodecl::AlignedFlag::make());
Nodecl::Reference reference = Nodecl::Reference::make(
Nodecl::ArraySubscript::make(
ogroup._subscripted.shallow_copy(),
Nodecl::List::make(
vload_index),
ogroup._basic_type),
ogroup._basic_type.get_pointer_to());
Nodecl::VectorAssignment vassignment =
Nodecl::VectorAssignment::make(
ogroup._registers[i].make_nodecl(true),
Nodecl::VectorLoad::make(
reference.shallow_copy(),
Utils::get_null_mask(),
flags,
ogroup._vector_type),
Utils::get_null_mask(),
Nodecl::NodeclBase::null(), 
ogroup._vector_type);
if (gen_init_prefetching)
{
Nodecl::ExpressionStatement prefetch_stmt =
Nodecl::ExpressionStatement::make(
Nodecl::VectorPrefetch::make(
reference.shallow_copy(),
const_value_to_nodecl(const_value_get_signed_int(PrefetchKind::L1_READ)),
reference.get_type()));
prefetching_list.append(prefetch_stmt);
}
Nodecl::ExpressionStatement exp_stmt =
Nodecl::ExpressionStatement::make(vassignment);
Optimizations::canonicalize_and_fold(exp_stmt, false );
result_list.append(exp_stmt);
}
result_list.prepend(prefetching_list);
return result_list;
}
Nodecl::NodeclBase OverlappedAccessesOptimizer::get_ogroup_iteration_update_pre(
const OverlapGroup& ogroup) const
{
const int size = ogroup._registers.size();
Nodecl::List flags;
if (ogroup._aligned_strategy)
flags = Nodecl::List::make(
Nodecl::AlignedFlag::make());
Nodecl::VectorAssignment vassignment =
Nodecl::VectorAssignment::make(
ogroup._registers[size-1].make_nodecl(true),
Nodecl::VectorLoad::make(
Nodecl::Reference::make(
Nodecl::ArraySubscript::make(
ogroup._subscripted.shallow_copy(),
Nodecl::List::make(
ogroup._registers_indexes[size-1].shallow_copy()),
ogroup._basic_type),
ogroup._basic_type.get_pointer_to()),
Utils::get_null_mask(),
flags,
ogroup._vector_type),
Utils::get_null_mask(),
Nodecl::NodeclBase::null(), 
ogroup._vector_type);
Nodecl::ExpressionStatement exp_stmt =
Nodecl::ExpressionStatement::make(vassignment);
Optimizations::canonicalize_and_fold(exp_stmt, false );
return exp_stmt;
}
Nodecl::List OverlappedAccessesOptimizer::get_ogroup_iteration_update_post(
const OverlapGroup& ogroup) const
{
Nodecl::List result_list;
const int size = ogroup._registers.size();
for(int i=0; i < (size-1); i++)
{
Nodecl::ExpressionStatement exp_stmt =
Nodecl::ExpressionStatement::make(
Nodecl::VectorAssignment::make(
ogroup._registers[i].make_nodecl(true),
ogroup._registers[i+1].make_nodecl(true),
Utils::get_null_mask(),
Nodecl::NodeclBase::null(), 
ogroup._registers[i].get_type()));
result_list.append(exp_stmt);
}
return result_list;
}
void OverlappedAccessesOptimizer::visit(const Nodecl::ForStatement& n)
{
Nodecl::ForStatement main_loop = n;
Nodecl::ForStatement if_epilog;
Nodecl::ForStatement last_epilog;
int min_unroll_factor = get_loop_min_unroll_factor(
main_loop);
std::cerr << "Min Unroll Factor for IF-EPILOG: " 
<< min_unroll_factor << std::endl;
if (min_unroll_factor > 0)
{
if_epilog = get_overlap_blocked_unrolled_loop(
main_loop, min_unroll_factor);
TL::HLT::LoopUnroll loop_unroller;
loop_unroller.set_loop(main_loop)
.set_unroll_factor(16)          
.unroll();
Nodecl::NodeclBase whole_main_transformation =
loop_unroller.get_whole_transformation();
main_loop = loop_unroller.get_unrolled_loop()
.as<Nodecl::ForStatement>();
last_epilog = loop_unroller.get_epilog_loop()
.as<Nodecl::ForStatement>();
n.replace(whole_main_transformation);
update_alignment_info(n, if_epilog);
last_epilog.prepend_sibling(if_epilog);
}
TL::Scope scope = main_loop.get_parent().get_parent().get_parent().
retrieve_context();
for(map_tlsym_objlist_int_t::const_iterator it = 
_environment._overlap_symbols_map.begin();
it != _environment._overlap_symbols_map.end();
it++)
{
TL::Symbol sym = it->first;
objlist_int_t overlap_params = it->second;
int min_group_loads = overlap_params[0];
int max_group_registers = overlap_params[1];
int max_groups = overlap_params[2];
objlist_nodecl_t main_loop_vector_loads =
get_adjacent_vector_loads_not_nested_in_for(
main_loop.get_statement(), sym);
const Nodecl::NodeclBase loop_ind_var = 
_analysis->get_linear_nodecls(main_loop).front(); 
const Nodecl::NodeclBase loop_ind_var_step = 
_analysis->get_linear_step(main_loop, loop_ind_var); 
if (!main_loop_vector_loads.empty())
{
objlist_ogroup_t overlap_groups = 
get_overlap_groups(
main_loop_vector_loads,
min_group_loads,
max_group_registers,
max_groups,
loop_ind_var,
loop_ind_var_step,
false );
int num_group = 0;
for(objlist_ogroup_t::iterator ogroup =
overlap_groups.begin();
ogroup != overlap_groups.end();
ogroup++)
{
ogroup->compute_basic_properties();
ogroup->compute_leftmost_rightmost_vloads(
_environment, max_group_registers);
retrieve_group_registers(*ogroup, scope, num_group);
insert_group_update_stmts(*ogroup, main_loop,
false );
replace_overlapped_loads(*ogroup, main_loop);
num_group++;
}
}
if (!if_epilog.is_null())
{
objlist_nodecl_t if_epilog_vector_loads =
get_adjacent_vector_loads_not_nested_in_for(
if_epilog.get_statement(), sym);
if (!if_epilog_vector_loads.empty())
{
objlist_ogroup_t if_epilog_overlap_groups = 
get_overlap_groups(
if_epilog_vector_loads,
min_group_loads,
max_group_registers,
max_groups,
loop_ind_var,
loop_ind_var_step,
false );
int num_group = 0;
for(objlist_ogroup_t::iterator ogroup =
if_epilog_overlap_groups.begin();
ogroup != if_epilog_overlap_groups.end();
ogroup++)
{
ogroup->compute_basic_properties();
ogroup->compute_leftmost_rightmost_vloads(
_environment, max_group_registers);
retrieve_group_registers(*ogroup, scope, num_group);
insert_group_update_stmts(*ogroup, if_epilog,
true );
replace_overlapped_loads(*ogroup, if_epilog);
num_group++;
}
}
}
}
if (min_unroll_factor > 0)
{
delete(_analysis);
_analysis = _first_analysis;
}
walk(main_loop.get_statement());
if (!if_epilog.is_null())
walk(if_epilog.get_statement());
if (min_unroll_factor > 1)
{
Nodecl::UnknownPragma unroll_pragma =
Nodecl::UnknownPragma::make("nounroll");
main_loop.prepend_sibling(unroll_pragma.shallow_copy());
last_epilog.prepend_sibling(unroll_pragma.shallow_copy());
}
if (min_unroll_factor == 16) 
{
Nodecl::Utils::remove_from_enclosing_list(if_epilog);
}
}
unsigned int OverlappedAccessesOptimizer::get_loop_min_unroll_factor(
Nodecl::ForStatement n)
{
const objlist_nodecl_t& ivs_list = _analysis->
get_linear_nodecls(n);
if (_environment._analysis_simd_scope == n)
return 0;
Nodecl::NodeclBase iv = ivs_list.front();
Nodecl::NodeclBase iv_step = 
_analysis->get_linear_step(n, iv);
if (Nodecl::Utils::structurally_equal_nodecls(iv,
_analysis->get_induction_variable_lower_bound(n, iv),
true))
return 0;
int unroll_factor = 0;
for(map_tlsym_objlist_int_t::const_iterator it = 
_environment._overlap_symbols_map.begin();
it != _environment._overlap_symbols_map.end();
it++)
{
TL::Symbol sym = it->first;
const int min_group_loads = it->second[0];
objlist_nodecl_t vector_loads =
get_adjacent_vector_loads_not_nested_in_for(
n.get_statement(), sym);
objlist_ogroup_t overlap_groups = 
get_overlap_groups(
vector_loads,
1, 
0, 
0, 
iv,
iv_step,
true );
for(objlist_ogroup_t::iterator ogroup =
overlap_groups.begin();
ogroup != overlap_groups.end();
ogroup++)
{
ogroup->compute_basic_properties();
const int ogroup_size = ogroup->_loads.size();
if(ogroup_size < min_group_loads)
{
if (ogroup->_inter_it_overlap)
{
if (ogroup_size <= min_group_loads &&
(unroll_factor * ogroup_size) < min_group_loads)
{
unroll_factor = min_group_loads /
ogroup_size; 
}
}
}
}
}
return unroll_factor;
}
Nodecl::ForStatement OverlappedAccessesOptimizer::
get_overlap_blocked_unrolled_loop(
const Nodecl::ForStatement& n,
const unsigned int block_size)
{
Nodecl::ForStatement blocked_unrolled_loop;
const objlist_nodecl_t& ivs_list = _analysis->
get_linear_nodecls(n);
if (block_size > 1 )
{
TL::HLT::LoopUnroll loop_unroller;
loop_unroller.set_loop(n)
.set_create_epilog(false)
.set_unroll_factor(block_size)
.unroll();
blocked_unrolled_loop = 
loop_unroller.get_unrolled_loop()
.as<Nodecl::ForStatement>();
}
else
{
blocked_unrolled_loop = n.shallow_copy()
.as<Nodecl::ForStatement>();
}
Nodecl::NodeclBase loop_header =
blocked_unrolled_loop.get_loop_header();
Nodecl::List loop_stmts =
blocked_unrolled_loop.get_statement()
.as<Nodecl::List>();
TL::LoopControlAdapter lc = 
TL::LoopControlAdapter(loop_header);
Nodecl::NodeclBase cond_node = lc.get_cond();
Nodecl::NodeclBase next_node = lc.get_next();
Nodecl::ExpressionStatement next_update_stmt =
Nodecl::ExpressionStatement::make(
next_node.shallow_copy());
Nodecl::NodeclBase if_statement_body = 
blocked_unrolled_loop.get_statement()
.as<Nodecl::List>().shallow_copy();
Nodecl::Utils::append_items_in_nested_compound_statement(
loop_stmts,
next_update_stmt.shallow_copy());
Nodecl::List outer_stmt = loop_stmts;
int num_unrolled_blocks = 
(_environment._vec_factor % block_size) == 0 ? 
(_environment._vec_factor / block_size) -1 :
_environment._vec_factor / block_size;
for (int i=1; i<num_unrolled_blocks; i++)
{
Nodecl::IfElseStatement if_else_stmt =
Nodecl::IfElseStatement::make(
cond_node.shallow_copy(),
if_statement_body.shallow_copy(),
Nodecl::NodeclBase::null());
objlist_nodecl_t vector_loads = Nodecl::Utils::
nodecl_get_all_nodecls_of_kind<Nodecl::VectorLoad>(if_else_stmt);
for (objlist_nodecl_t::const_iterator vl = vector_loads.begin();
vl != vector_loads.end();
vl++)
{
bool found = false;
for (map_tlsym_objlist_int_t::const_iterator overlap_symbol =
_environment._overlap_symbols_map.begin();
overlap_symbol != _environment._overlap_symbols_map.end();
overlap_symbol++)
{
if ((overlap_symbol->first) == Utils::get_vector_load_subscripted(
vl->as<Nodecl::VectorLoad>()).get_symbol())
{
for (objlist_nodecl_t::const_iterator iv =
ivs_list.begin();
iv != ivs_list.end();
iv++)
{
Nodecl::Add iv_plus_boffset =
Nodecl::Add::make(
iv->shallow_copy(),
const_value_to_nodecl(
const_value_get_signed_int(i * block_size)),
TL::Type::get_int_type());
Nodecl::Utils::nodecl_replace_nodecl_by_structure(
*vl,
*iv,
iv_plus_boffset);
}
found = true;
}
}
ERROR_CONDITION(!found, "Overlap: This code is not going to work without local IV increment", 0);
}
Nodecl::Utils::append_items_in_nested_compound_statement(
if_else_stmt.get_then(),
next_update_stmt.shallow_copy());
Nodecl::Utils::append_items_in_nested_compound_statement(
outer_stmt, if_else_stmt);
outer_stmt = if_else_stmt.get_then().as<Nodecl::List>();
}
blocked_unrolled_loop.replace(
Nodecl::IfElseStatement::make(
cond_node.shallow_copy(),
blocked_unrolled_loop.get_statement().shallow_copy(),
Nodecl::NodeclBase::null()));
return blocked_unrolled_loop;
}
objlist_nodecl_t OverlappedAccessesOptimizer::
get_adjacent_vector_loads_not_nested_in_for(
const Nodecl::NodeclBase& n,
const TL::Symbol& sym)
{
objlist_nodecl_t result;
objlist_nodecl_t vector_loads = Nodecl::Utils::
nodecl_get_all_nodecls_of_kind<Nodecl::VectorLoad>(n);
objlist_nodecl_t nested_for_stmts = Nodecl::Utils::
nodecl_get_all_nodecls_of_kind<Nodecl::ForStatement>(n);
for(objlist_nodecl_t::iterator vload = vector_loads.begin();
vload != vector_loads.end();
vload++)
{
bool vload_is_nested_in_nested_for = false;
for(objlist_nodecl_t::iterator nested_for = nested_for_stmts.begin();
nested_for != nested_for_stmts.end();
nested_for++)
{
if (Nodecl::Utils::nodecl_contains_nodecl_by_pointer(
*nested_for, *vload))
{
vload_is_nested_in_nested_for = true;
break;
}
}
if (!vload_is_nested_in_nested_for)
{
Nodecl::NodeclBase subscripted= 
Utils::get_vector_load_subscripted(
vload->as<Nodecl::VectorLoad>());
if (subscripted.is<Nodecl::Symbol>() &&
(subscripted.get_symbol() == sym))
{
result.append(*vload);
}
}
}
return result;
}
void OverlappedAccessesOptimizer::retrieve_group_registers(
OverlapGroup& ogroup,
TL::Scope& scope,
const int num_group)
{ 
Nodecl::NodeclBase leftmost_index = 
Utils::get_vector_load_subscript(ogroup._leftmost_group_vload);
int vec_factor = ogroup._vector_type.vector_num_elements();
for (int i=0; i<ogroup._num_registers; i++)
{
std::stringstream new_sym_name;
new_sym_name << "__overlap_" 
<< Utils::get_subscripted_symbol(ogroup._subscripted).get_name() << "_"
<< num_group << "_"
<< i;
if (!scope.get_symbol_from_name(
new_sym_name.str()).is_valid())
{
TL::Symbol new_sym = scope.new_symbol(new_sym_name.str());
new_sym.get_internal_symbol()->kind = SK_VARIABLE;
symbol_entity_specs_set_is_user_declared(new_sym.get_internal_symbol(), 1);
new_sym.set_type(ogroup._vector_type);
ogroup._registers.push_back(new_sym);
}
else
{
TL::Symbol sym = 
scope.get_symbol_from_name(new_sym_name.str());
ERROR_CONDITION(!sym.is_valid(), "cache symbol is invalid.", 0);
ogroup._registers.push_back(sym);
}
Nodecl::NodeclBase new_reg_index;
if (i == 0)
{
new_reg_index = leftmost_index;
}
else
{
new_reg_index = Nodecl::Add::make(
leftmost_index.shallow_copy(),
const_value_to_nodecl(const_value_mul(
const_value_get_signed_int(i),
const_value_get_signed_int(
vec_factor))),
leftmost_index.get_type());
Optimizations::canonicalize_and_fold(
new_reg_index, false );
}
ogroup._registers_indexes.push_back(
new_reg_index);
}
}
void OverlappedAccessesOptimizer::insert_group_update_stmts(
OverlapGroup& ogroup,
const Nodecl::ForStatement& n,
const bool is_overlap_epilog)
{
bool is_simd_loop = _environment._analysis_simd_scope == n;
if (ogroup._inter_it_overlap)
{
if (need_init_cache(!is_simd_loop, _is_simd_epilog,
is_overlap_epilog))
{
Nodecl::NodeclBase init_stmts =
get_ogroup_init_statements(ogroup, n, 
is_simd_loop, _is_omp_simd_for);
if(is_simd_loop)
{
_prependix_stmts.prepend(init_stmts);
}
else
{
n.prepend_sibling(init_stmts);
}
}
if (need_update_post(!is_simd_loop, _is_simd_epilog,
is_overlap_epilog))
{
if (!_in_place)
{
Nodecl::List post_stmts = 
get_ogroup_iteration_update_post(ogroup);
Nodecl::Utils::append_items_in_nested_compound_statement(
n.get_statement(), post_stmts);
}
}
if (!_in_place)
{
Nodecl::NodeclBase pre_stmt = get_ogroup_iteration_update_pre(ogroup);
Nodecl::Utils::prepend_items_in_nested_compound_statement(
n.get_statement(), pre_stmt);
}
}
else 
{
Nodecl::NodeclBase init_stmts =
get_ogroup_init_statements(ogroup, n,
is_simd_loop, _is_omp_simd_for);
Nodecl::Utils::prepend_items_in_nested_compound_statement(
n.get_statement(), init_stmts);
ERROR_CONDITION(_in_place, "intra-iteration in place not implemented\n", 0);
}
}
void OverlappedAccessesOptimizer::replace_overlapped_loads(
OverlapGroup& ogroup,
const Nodecl::NodeclBase& nesting_node)
{
for(const auto& load_it : ogroup._loads)
{
Nodecl::NodeclBase load_subscript =
Utils::get_vector_load_subscript(
load_it.as<Nodecl::VectorLoad>());
Nodecl::Minus shifted_elements = Nodecl::Minus::make(
load_subscript.shallow_copy(),
ogroup._registers_indexes[0].no_conv().shallow_copy(),
load_subscript.get_type());
TL::Optimizations::UnitaryReductor unitary_reductor;
unitary_reductor.reduce(shifted_elements);
if (shifted_elements.is_constant())
{
const_value_t* mod = const_value_mod(
shifted_elements.get_constant(),
const_value_get_signed_int(
_environment._vec_factor));
const_value_t* div = const_value_div(
shifted_elements.get_constant(),
const_value_get_signed_int(
_environment._vec_factor));
int first_register = const_value_cast_to_signed_int(div);
int final_offset = const_value_cast_to_signed_int(mod);
bool uses_last_register = false;
if (const_value_is_zero(mod))
{
load_it.replace(
ogroup._registers[first_register].
make_nodecl(true));
if (first_register == ((int) ogroup._registers.size()-1))
uses_last_register = true;
}
else
{
load_it.replace(Nodecl::VectorAlignRight::make(
ogroup._registers[first_register+1].make_nodecl(true),
ogroup._registers[first_register].make_nodecl(true),
const_value_to_nodecl(const_value_get_signed_int(final_offset)),
load_it.as<Nodecl::VectorLoad>().
get_mask().shallow_copy(),
ogroup._registers[first_register].get_type()));
if ((first_register+1) == ((int) ogroup._registers.size()-1))
uses_last_register = true;
}
if (_in_place && uses_last_register && ogroup._inter_it_overlap && !ogroup._is_set_in_place_update_pre)
{
Nodecl::NodeclBase pre_stmt = get_ogroup_iteration_update_pre(ogroup);
Nodecl::Utils::prepend_sibling_statement(load_it, pre_stmt, nesting_node  );
ogroup._is_set_in_place_update_pre = true;
}
}
else
{
}
}
if (_in_place && ogroup._inter_it_overlap)
{
Nodecl::NodeclBase post_stmt = get_ogroup_iteration_update_post(ogroup);
Nodecl::Utils::append_sibling_statement(ogroup._loads.back(), post_stmt, nesting_node  );
}
}
}
}
