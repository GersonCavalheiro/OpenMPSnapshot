#include <queue>
#include "cxx-diagnostic.h"
#include "tl-analysis-utils.hpp"
#include "tl-datareference.hpp"
#include "tl-induction-variables-data.hpp"
#include "tl-optimizations.hpp"
#include "tl-oss-lint.hpp"
#include "tl-ranges-common.hpp"
namespace TL {
namespace Nanos6 {
std::set<Analysis::ExtensibleGraph*> analyzed_pcfgs;
std::set<Analysis::Node*> analyzed_tasks;
OssLint::OssLint()
: _only_tasks_str(""), _only_tasks(false), _pcfgs()
{
set_phase_name("OmpSs-2 Lint");
set_phase_description("This phase is intended to lighten the runtime lint tool for OmpSs-2 by means of static analysis");
register_parameter("oss_lint_only_tasks",
"OmpSs Lint phase only analyzes tasks",
_only_tasks_str,
"0").connect(std::bind(&OssLint::set_only_tasks_mode, this, std::placeholders::_1));
}
void OssLint::set_only_tasks_mode(const std::string& oss_lint_only_tasks_str)
{
if( oss_lint_only_tasks_str == "1")
_only_tasks = true;
}
void OssLint::run(TL::DTO& dto)
{
if (VERBOSE)
{
std::cerr << "===========================================" << std::endl;
std::cerr << "OSS-LINT_ Executing analysis required for OmpSs-2 correctness checking" << std::endl;
}
Nodecl::NodeclBase top_level = *std::static_pointer_cast<Nodecl::NodeclBase>(dto["nodecl"]);
if (_only_tasks
&& !Nodecl::Utils::nodecl_contains_nodecl_of_kind<Nodecl::OpenMP::Task>(top_level)
&& !Nodecl::Utils::nodecl_contains_nodecl_of_kind<Nodecl::OmpSs::TaskCall>(top_level))
{
if (VERBOSE)
{
std::cerr << " Skipping file " << top_level.get_filename() << " because it does not contain tasks" << std::endl;
std::cerr << "===========================================" << std::endl;
}
return;
}
TL::Analysis::AnalysisBase analysis( true);
analysis.dominator_tree(top_level);
analysis.induction_variables(top_level,  true);
_pcfgs = analysis.get_pcfgs();
for (ObjectList<Analysis::ExtensibleGraph*>::iterator it = _pcfgs.begin(); it != _pcfgs.end(); ++it)
analysis.print_pcfg((*it)->get_name());
std::cerr << "================================================================" << std::endl;
std::cerr << "= launch_correctness :  " << top_level.get_filename() << std::endl;
launch_correctness(analysis, top_level.get_filename());
if (VERBOSE)
{
std::cerr << "================================================================" << std::endl;
}
}
void OssLint::launch_correctness(
const TL::Analysis::AnalysisBase& analysis,
std::string current_filename)
{
std::queue<Analysis::DomTreeNode*> dtnodes_to_treat;
std::set<Analysis::DomTreeNode*> dtnodes_considered;
Analysis::DominatorTree* dt = analysis.get_dom_tree();
std::set<Analysis::DomTreeNode*> leafs = dt->get_leafs();
for (std::set<Analysis::DomTreeNode*>::iterator it = leafs.begin();
it != leafs.end(); ++it)
{
dtnodes_to_treat.push(*it);
dtnodes_considered.insert(*it);
}
while (!dtnodes_to_treat.empty())
{
Analysis::DomTreeNode* dtn = dtnodes_to_treat.front();
dtnodes_to_treat.pop();
std::set<Analysis::DomTreeNode*> pred = dtn->get_predecessors();
for (std::set<Analysis::DomTreeNode*>::iterator itp = pred.begin(); itp != pred.end(); ++itp)
{
if (dtnodes_considered.find(*itp) == dtnodes_considered.end())
{
dtnodes_to_treat.push(*itp);
dtnodes_considered.insert(*itp);
}
}
Symbol dtn_sym = dtn->get_function_symbol();
if (!dt->get_function_symbols().contains(dtn_sym)
|| dtn_sym.is_lambda())
continue;
std::string func_name = dtn_sym.get_name();
std::cerr << "============ Function: " << func_name << std::endl;
TL::Analysis::ExtensibleGraph* pcfg = analysis.get_pcfg_by_func_name(func_name);
ERROR_CONDITION(pcfg==NULL,
"PCFG for function '%s' not found!\n",
func_name.c_str());
if (_only_tasks)
oss_lint_analyze_tasks(pcfg);
else
oss_lint_analyze_all(pcfg);
}
}
namespace {
std::map<Analysis::Node*, LintAttributes*> _pending_attrs;
bool all_symbols_alive_rec(const Analysis::NodeclSet& var_set, Scope sc)
{
if (var_set.empty())
return true;
for (Analysis::NodeclSet::iterator it = var_set.begin(); it != var_set.end(); ++it)
{
ObjectList<Symbol> syms = Nodecl::Utils::get_all_symbols(*it);
for (ObjectList<Symbol>::iterator its = syms.begin(); its != syms.end(); ++its)
{
if (its->is_member())
continue;
if (!sc.scope_is_enclosed_by(its->get_scope()) && sc != its->get_scope())
{
return false;
}
}
}
return true;
}
void append_to_path(
ObjectList<Nodecl::NodeclBase>& path,
const Nodecl::NodeclBase& n)
{
Nodecl::NodeclBase parent = n.get_parent();
while (!parent.is_null())
{
if (parent.is<Nodecl::ForStatement>())
{   
Nodecl::NodeclBase header = parent.as<Nodecl::ForStatement>().get_loop_header();
if (Nodecl::Utils::nodecl_contains_nodecl_by_pointer( header,  n))
return;
}
parent = parent.get_parent();
}
for (ObjectList<Nodecl::NodeclBase>::iterator it = path.begin(); it != path.end(); )
{
if (Nodecl::Utils::nodecl_contains_nodecl_by_pointer(n, *it))
{
it = path.erase(it);
}
else
{
++it;
}
}
std::string path_str;
for (ObjectList<Nodecl::NodeclBase>::const_iterator it = path.begin();
it != path.end(); )
{
path_str += it->prettyprint();
++it;
if (it != path.end())
path_str += "\n ";
}
path.append(n);
}
void append_to_attrs(LintAttributes* attrs, const Nodecl::List& env)
{
Nodecl::NodeclBase dep_out_clause = env.find_first<Nodecl::OpenMP::DepOut>();
Analysis::NodeclSet dep_out_set;
if (!dep_out_clause.is_null())
{
Nodecl::List dep_out_list = dep_out_clause.as<Nodecl::OpenMP::DepOut>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_out_list.begin(); it != dep_out_list.end(); ++it)
{
Nodecl::NodeclBase v = it->shallow_copy();
TL::Optimizations::canonicalize_and_fold(v,  false);
attrs->add_out(v);
}
}
Nodecl::NodeclBase dep_in_clause = env.find_first<Nodecl::OpenMP::DepIn>();
if (!dep_in_clause.is_null())
{
Nodecl::List dep_in_list = dep_in_clause.as<Nodecl::OpenMP::DepIn>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_in_list.begin(); it != dep_in_list.end(); ++it)
{
Nodecl::NodeclBase v = it->shallow_copy();
TL::Optimizations::canonicalize_and_fold(v,  false);
attrs->add_in(v);
}
}
Nodecl::NodeclBase dep_inout_clause = env.find_first<Nodecl::OpenMP::DepInout>();
Analysis::NodeclSet dep_inout_set;
if (!dep_inout_clause.is_null())
{
Nodecl::List dep_inout_list = dep_inout_clause.as<Nodecl::OpenMP::DepInout>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_inout_list.begin(); it != dep_inout_list.end(); ++it)
{
Nodecl::NodeclBase v = it->shallow_copy();
TL::Optimizations::canonicalize_and_fold(v,  false);
attrs->add_inout(v);
}
}
Nodecl::NodeclBase lint_alloc_clause = env.find_first<Nodecl::OmpSs::LintAlloc>();
Analysis::NodeclSet lint_alloca_set;
if (!lint_alloc_clause.is_null())
{
Nodecl::List lint_alloc_list = lint_alloc_clause.as<Nodecl::OmpSs::LintAlloc>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = lint_alloc_list.begin(); it != lint_alloc_list.end(); ++it)
attrs->add_alloc(*it);
}
Nodecl::NodeclBase lint_free_clause = env.find_first<Nodecl::OmpSs::LintFree>();
Analysis::NodeclSet lint_free_set;
if (!lint_free_clause.is_null())
{
Nodecl::List lint_free_list = lint_free_clause.as<Nodecl::OmpSs::LintFree>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = lint_free_list.begin(); it != lint_free_list.end(); ++it)
attrs->add_free(*it);
}
}
bool is_statement(const Nodecl::NodeclBase& n)
{
return n.is<Nodecl::IfElseStatement>() || n.is<Nodecl::GotoStatement>()
|| n.is<Nodecl::SourceComment>() || n.is<Nodecl::SwitchStatement>()
|| n.is<Nodecl::EmptyStatement>() || n.is<Nodecl::LabeledStatement>()
|| n.is<Nodecl::FunctionCode>() || n.is<Nodecl::ExpressionStatement>()
|| n.is<Nodecl::Text>() || n.is<Nodecl::ReturnStatement>()
|| n.is<Nodecl::CaseStatement>() || n.is<Nodecl::ContinueStatement>()
|| n.is<Nodecl::Context>() || n.is<Nodecl::BreakStatement>()
|| n.is<Nodecl::DefaultStatement>() || n.is<Nodecl::TryBlock>()
|| n.is<Nodecl::WhileStatement>() || n.is<Nodecl::DoStatement>()
|| n.is<Nodecl::CatchHandler>() || n.is<Nodecl::ErrStatement>()
|| n.is<Nodecl::ForStatement>() || n.is<Nodecl::ObjectInit>()
|| n.is<Nodecl::CompoundStatement>()
|| n.is<Nodecl::OpenMP::FlushMemory>() || n.is<Nodecl::OpenMP::Sections>()
|| n.is<Nodecl::OpenMP::Master>() || n.is<Nodecl::OpenMP::Critical>()
|| n.is<Nodecl::OpenMP::Taskyield>() || n.is<Nodecl::OpenMP::Teams>()
|| n.is<Nodecl::OpenMP::TargetData>() || n.is<Nodecl::OpenMP::Workshare>()
|| n.is<Nodecl::OpenMP::SimdFor>() || n.is<Nodecl::OpenMP::Distribute>()
|| n.is<Nodecl::OpenMP::Distribute>() || n.is<Nodecl::OpenMP::Single>()
|| n.is<Nodecl::OpenMP::Parallel>() || n.is<Nodecl::OpenMP::BarrierSignal>()
|| n.is<Nodecl::OpenMP::TargetTaskUndeferred>() || n.is<Nodecl::OpenMP::ForAppendix>()
|| n.is<Nodecl::OpenMP::SimdReduction>() || n.is<Nodecl::OpenMP::ParallelSimdFor>()
|| n.is<Nodecl::OpenMP::Taskloop>() || n.is<Nodecl::OpenMP::Taskgroup>()
|| n.is<Nodecl::OpenMP::Atomic>() || n.is<Nodecl::OpenMP::TargetUpdate>()
|| n.is<Nodecl::OpenMP::Taskwait>() || n.is<Nodecl::OpenMP::BarrierWait>()
|| n.is<Nodecl::OpenMP::BarrierFull>() || n.is<Nodecl::TemplateFunctionCode>()
|| n.is<Nodecl::OpenMP::For>() || n.is<Nodecl::OpenMP::Task>()
|| n.is<Nodecl::OmpSs::Lint>() || n.is<Nodecl::OmpSs::Unregister>()
|| n.is<Nodecl::OmpSs::Release>() || n.is<Nodecl::OmpSs::Register>();
}
Nodecl::NodeclBase get_enclosing_statement(const Nodecl::NodeclBase& n)
{
Nodecl::NodeclBase enclosing_stmt = n;
while (!is_statement(enclosing_stmt))
{
enclosing_stmt = enclosing_stmt.get_parent();
}
return enclosing_stmt;
}
bool something_to_propagate(ObjectList<Nodecl::NodeclBase> stmts)
{
if (stmts.empty())
return false;
if (stmts.size() == 1)
{
if (stmts[0].is<Nodecl::OpenMP::Task>()
|| stmts[0].is<Nodecl::OmpSs::Lint>())
return false;
if (stmts[0].is<Nodecl::Context>())
{
Nodecl::List l = stmts[0].as<Nodecl::Context>().get_in_context().as<Nodecl::List>().at(0).as<Nodecl::CompoundStatement>().get_statements().as<Nodecl::List>();
for (Nodecl::List::iterator it = l.begin(); it != l.end(); ++it)
{
if (!it->is<Nodecl::OpenMP::Task>() && !it->is<Nodecl::CxxDef>())
return true;
}
return false;
}
}
return true;
}
bool replace_ivs_with_ranges(
Nodecl::NodeclBase& v,
const Analysis::Utils::InductionVarList& ind_vars)
{
bool replacement_done = false;
for (Analysis::Utils::InductionVarList::const_iterator it_iv = ind_vars.begin();
it_iv != ind_vars.end(); ++it_iv)
{
ERROR_CONDITION((*it_iv)->get_lb().size()>1 || (*it_iv)->get_ub().size()>1,
"Induction variable '%s' has multiple lower and/or upper bounds. Lint does not yet support this feature.",
(*it_iv)->get_variable().prettyprint().c_str());
Nodecl::NodeclBase iv = (*it_iv)->get_variable();
Nodecl::NodeclBase lb = Nodecl::Utils::deep_copy(*((*it_iv)->get_lb().begin()), iv.retrieve_context());
Nodecl::NodeclBase ub = Nodecl::Utils::deep_copy(*((*it_iv)->get_ub().begin()), iv.retrieve_context());
Nodecl::NodeclBase incr = Nodecl::Utils::deep_copy((*it_iv)->get_increment(), iv.retrieve_context());
Nodecl::Range iv_range =
Nodecl::Range::make(
lb, ub, incr,
Type::get_int_type());
Nodecl::Utils::CollectStructuralNodeFinderVisitor finder_visitor( iv);
finder_visitor.generic_finder( v);
ObjectList<Nodecl::NodeclBase> found_nodes = finder_visitor._found_nodes;
for (ObjectList<Nodecl::NodeclBase>::iterator itf = found_nodes.begin(); itf != found_nodes.end(); ++itf)
{
bool iv_is_simple_subscript = true;
Nodecl::NodeclBase parent = itf->get_parent();
while (iv_is_simple_subscript && !parent.is<Nodecl::ArraySubscript>())
{
if (!parent.is<Nodecl::ArraySubscript>() && !parent.is<Nodecl::List>() && !parent.is<Nodecl::Conversion>())
{
iv_is_simple_subscript = false;
}
parent = parent.get_parent();
}
if (iv_is_simple_subscript)
{
Nodecl::ArraySubscript arr =
Nodecl::Utils::get_enclosing_list(itf->no_conv()).get_parent().as<Nodecl::ArraySubscript>();
Nodecl::List subscripts = arr.get_subscripts().as<Nodecl::List>();
for (Nodecl::List::iterator its = subscripts.begin(); its != subscripts.end(); ++its)
{
if (Nodecl::Utils::structurally_equal_nodecls(*its, iv,  true))
{
its->replace(iv_range.shallow_copy());
replacement_done = true;
}
}
}
else
{
parent = itf->get_parent();
while (!parent.is_null() && !parent.is<Nodecl::ArraySubscript>())
{
if (parent.is<Nodecl::Range>())
{
Nodecl::Range r = parent.as<Nodecl::Range>();
if (Nodecl::Utils::nodecl_contains_nodecl_by_structure(
r.get_upper(), iv))
{
Nodecl::Utils::nodecl_replace_nodecl_by_structure(
r.get_upper(),
iv,
iv_range.get_upper().shallow_copy());
replacement_done = true;
}
if (Nodecl::Utils::nodecl_contains_nodecl_by_structure(
r.get_lower(), iv))
{
Nodecl::Utils::nodecl_replace_nodecl_by_structure(
r.get_lower(),
iv,
iv_range.get_lower().shallow_copy());
replacement_done = true;
}
}
parent = parent.get_parent();
}
if (!replacement_done)
{
Nodecl::Utils::nodecl_replace_nodecl_by_structure( v,  iv,  iv_range.shallow_copy());
replacement_done = true;
}
}
}
}
Analysis::Utils::reduce_expression(v);
return replacement_done;
}
bool update_attributes(Analysis::Node* n,
LintAttributes* attrs,
const Analysis::Utils::InductionVarList& ind_vars)
{
const Analysis::NodeclSet& ue_vars = n->get_ue_vars();
const Analysis::NodeclSet& killed_vars = n->get_killed_vars();
Analysis::NodeclSet new_in, new_out, new_inout;
for (Analysis::NodeclSet::iterator it = ue_vars.begin();
it != ue_vars.end(); ++it)
{
if (it->prettyprint()=="::std::cout")
continue;
if (it->get_type().no_ref().is_const() && !it->is<Nodecl::ArraySubscript>())
continue;
if (it->get_symbol().is_valid() && it->get_symbol().is_function())
continue;   
Nodecl::NodeclBase v = Nodecl::Utils::deep_copy(*it, it->retrieve_context());
if (v.is<Nodecl::ArraySubscript>())
{
Nodecl::List subscripts = v.as<Nodecl::ArraySubscript>().get_subscripts().as<Nodecl::List>();
if (Nodecl::Utils::nodecl_contains_nodecl_of_kind<Nodecl::ArraySubscript>(subscripts))
return false;
replace_ivs_with_ranges(v,  ind_vars);
}
if (killed_vars.find(*it) == killed_vars.end())
{
TL::Optimizations::canonicalize_and_fold(v,  false);
new_in.insert(v);
}
else
{
TL::Optimizations::canonicalize_and_fold(v,  false);
new_inout.insert(v);
}
}
bool verified = true;
for (Analysis::NodeclSet::iterator it = killed_vars.begin();
it != killed_vars.end(); ++it)
{
if (ue_vars.find(*it) != ue_vars.end())
continue;
Nodecl::NodeclBase v = Nodecl::Utils::deep_copy(*it, it->retrieve_context());
if (v.is<Nodecl::ArraySubscript>())
{
Nodecl::List subscripts = v.as<Nodecl::ArraySubscript>().get_subscripts().as<Nodecl::List>();
if (Nodecl::Utils::nodecl_contains_nodecl_of_kind<Nodecl::ArraySubscript>(subscripts))
return false;
replace_ivs_with_ranges(v,  ind_vars);
}
TL::Optimizations::canonicalize_and_fold(v,  false);
new_out.insert(v);
}
for (Analysis::NodeclSet::iterator it = new_in.begin(); it != new_in.end(); ++it)
attrs->add_in(*it);
for (Analysis::NodeclSet::iterator it = new_out.begin(); it != new_out.end(); ++it)
attrs->add_out(*it);
for (Analysis::NodeclSet::iterator it = new_inout.begin(); it != new_inout.end(); ++it)
attrs->add_inout(*it);
return verified;
}
std::string print_list(const Analysis::NodeclSet& dep_set)
{
std::string dep_str = "";
for (Analysis::NodeclSet::iterator it = dep_set.begin(); it != dep_set.end(); )
{
dep_str += it->prettyprint();
++it;
if (it != dep_set.end())
dep_str += ",";
}
return dep_str;
}
}
void OssLint::wrap_statements(ObjectList<Nodecl::NodeclBase>& stmts,
LintAttributes* attrs)
{
ERROR_CONDITION(stmts.empty() && (!attrs->is_empty()),
"Lint: Wrapping empty list of statements with non null attributes.\n", 0);
if (!something_to_propagate(stmts))
{
stmts.clear();
attrs->clear();
return;
}
if (attrs->is_empty())
{
stmts.clear();
return;
}
if (stmts.size() == 1
&& (stmts.at(0).is<Nodecl::OmpSs::Lint>()
|| (stmts.at(0).is<Nodecl::OmpSs::TaskCall>()
&& !stmts.at(0).as<Nodecl::OmpSs::TaskCall>().get_environment().as<Nodecl::List>()
.find_first<Nodecl::OmpSs::LintVerified>().is_null())))
{   
attrs->clear();
stmts.clear();
return;
}
Nodecl::List env;
Analysis::NodeclSet& in = attrs->get_in();
Analysis::NodeclSet& out = attrs->get_out();
Analysis::NodeclSet& inout = attrs->get_inout();
Analysis::NodeclSet& lint_alloc = attrs->get_alloc();
Analysis::NodeclSet& lint_free = attrs->get_free();
for (Analysis::NodeclSet::iterator iti = in.begin(); iti != in.end(); )
{
Analysis::NodeclSet::iterator ito = out.find(*iti);
if (ito != out.end())
{
inout.insert(*iti);
out.erase(ito);
iti = in.erase(iti);
}
else
{
++iti;
}
}
for (Analysis::NodeclSet::iterator itio = inout.begin(); itio != inout.end(); ++itio)
{
Analysis::NodeclSet::iterator ito = out.find(*itio);
if (ito != out.end())
{
out.erase(ito);
}
Analysis::NodeclSet::iterator iti = in.find(*itio);
if (iti != in.end())
{
in.erase(iti);
}
}
std::string dependencies_str = "";
if (!in.empty())
{
ObjectList<Nodecl::NodeclBase> in_list(in.begin(), in.end());
env.append(
Nodecl::OpenMP::DepIn::make(
Nodecl::List::make(in_list)
)
);
dependencies_str += "depend(in:" + print_list(in) + ") ";
}
if (!out.empty())
{
ObjectList<Nodecl::NodeclBase> out_list(out.begin(), out.end());
env.append(
Nodecl::OpenMP::DepOut::make(
Nodecl::List::make(out_list)
)
);
dependencies_str += "depend(out:" + print_list(out) + ") ";
}
if (!inout.empty())
{
ObjectList<Nodecl::NodeclBase> inout_list(inout.begin(), inout.end());
env.append(
Nodecl::OpenMP::DepInout::make(
Nodecl::List::make(inout_list)
)
);
dependencies_str += "depend(inout:" + print_list(inout) + ") ";
}
if (!lint_alloc.empty())
{
ObjectList<Nodecl::NodeclBase> lint_alloc_list(lint_alloc.begin(), lint_alloc.end());
env.append(
Nodecl::OmpSs::LintAlloc::make(
Nodecl::List::make(lint_alloc_list)
)
);
dependencies_str += "alloc(" + print_list(lint_alloc) + ") ";
}
if (!lint_free.empty())
{
ObjectList<Nodecl::NodeclBase> lint_free_list(lint_free.begin(), lint_free.end());
env.append(
Nodecl::OmpSs::LintFree::make(
Nodecl::List::make(lint_free_list)
)
);
dependencies_str += "free(" + print_list(lint_alloc) + ") ";
}
ObjectList<Nodecl::NodeclBase> real_stmts;
if (stmts.size() == 1 && stmts.at(0).is<Nodecl::Context>())
{
Nodecl::NodeclBase in_ctx_list_elem = stmts.at(0).as<Nodecl::Context>().get_in_context().as<Nodecl::List>().front();
if (in_ctx_list_elem.is<Nodecl::CompoundStatement>())
{
real_stmts = in_ctx_list_elem.as<Nodecl::CompoundStatement>().get_statements().as<Nodecl::List>().to_object_list();
}
else
{   
real_stmts.append(in_ctx_list_elem);
}
}
else
{
Nodecl::NodeclBase parent;
for (ObjectList<Nodecl::NodeclBase>::const_iterator it = stmts.begin();
it != stmts.end(); ++it)
{
Nodecl::NodeclBase stmt;
if (is_statement(*it))
{
stmt = *it;
}
else
{   
stmt = get_enclosing_statement(*it);
}
real_stmts.prepend(stmt);
}
}
std::string stmts_str = "";
for (ObjectList<Nodecl::NodeclBase>::iterator it = real_stmts.begin();
it != real_stmts.end(); ++it)
stmts_str += it->prettyprint();
std::cerr << "lint-debug: wrapping statements between lines "
<< real_stmts.front().get_line()
<< " and " << real_stmts.back().get_line()
<< " with a \"#pragma oss lint: " << dependencies_str
<< "\"" << std::endl;
Nodecl::List stmts_list;
for (ObjectList<Nodecl::NodeclBase>::iterator it = real_stmts.begin();
it != real_stmts.end(); ++it)
stmts_list.append(it->shallow_copy());
Nodecl::List enclosing_list = Nodecl::Utils::get_enclosing_list(real_stmts.at(0)).as<Nodecl::List>();
Scope sc = enclosing_list.retrieve_context();
Nodecl::NodeclBase lint = 
Nodecl::OmpSs::Lint::make(
env,
Nodecl::List::make(
Nodecl::Context::make(
Nodecl::List::make(
Nodecl::CompoundStatement::make(
stmts_list,
Nodecl::NodeclBase::null()
)
),
sc.get_decl_context()
)
)
);
for (Nodecl::List::iterator itp = enclosing_list.begin();
itp != enclosing_list.end(); ++itp)
{
if (Nodecl::Utils::structurally_equal_nodecls(real_stmts.front(), *itp,  true))
{
enclosing_list.insert(itp, lint);
break;
}
}
for (ObjectList<Nodecl::NodeclBase>::iterator it = real_stmts.begin();
it != real_stmts.end(); ++it)
{
for (Nodecl::List::iterator itp = enclosing_list.begin();
itp != enclosing_list.end(); ++itp)
{
if (Nodecl::Utils::structurally_equal_nodecls(*itp, *it,  true))
{
enclosing_list.erase(itp);
break;
}
}
}
attrs->clear();
stmts.clear();
}
static bool check_attributes_list(
const Analysis::NodeclSet& previous_all_in,
const Analysis::NodeclSet& new_all_out)
{
bool defines_previous_variables = false;
for (Analysis::NodeclSet::iterator iti = previous_all_in.begin();
iti != previous_all_in.end() && !defines_previous_variables; ++iti)
{
Nodecl::NodeclBase in = iti->no_conv().shallow_copy();
while (in.is<Nodecl::ArraySubscript>())
in = in.as<Nodecl::ArraySubscript>().get_subscripted().no_conv();
for (Analysis::NodeclSet::iterator ito = new_all_out.begin();
ito != new_all_out.end() && !defines_previous_variables; ++ito)
{
Nodecl::NodeclBase out = ito->no_conv().shallow_copy();
while (out.is<Nodecl::ArraySubscript>())
out = out.as<Nodecl::ArraySubscript>().get_subscripted().no_conv();
if (Nodecl::Utils::structurally_equal_nodecls(in, out, true))
defines_previous_variables = true;
}
}
return defines_previous_variables;
}
static void gather_arrays_info(
const Analysis::NodeclSet& source,
Analysis::NodeclSet& target,
bool gather_subscripted)
{
for (Analysis::NodeclSet::iterator it = source.begin(); it != source.end(); ++it)
{
Nodecl::NodeclBase v = it->no_conv();
if (v.is<Nodecl::ArraySubscript>())
{   
Nodecl::ArraySubscript arr = v.as<Nodecl::ArraySubscript>();
ObjectList<Nodecl::NodeclBase> mem_accesses = Nodecl::Utils::get_all_memory_accesses(arr.get_subscripts());
target.insert(mem_accesses.begin(), mem_accesses.end());
if (gather_subscripted)
target.insert(arr.get_subscripted());
}
else if (v.get_type().is_pointer())
{
target.insert(v);
}
}
}
static bool new_attributes_define_previous_attributes(
LintAttributes* old_attrs,
LintAttributes* new_attrs)
{
Analysis::NodeclSet previous_all_in;
gather_arrays_info(old_attrs->get_in(), previous_all_in, true);
gather_arrays_info(old_attrs->get_out(), previous_all_in, false);
gather_arrays_info(old_attrs->get_inout(), previous_all_in, true);
Analysis::NodeclSet new_all_outs = Analysis::Utils::nodecl_set_union(new_attrs->get_out(), new_attrs->get_inout());
return check_attributes_list(previous_all_in, new_all_outs);
}
static void get_dependency_sets(
const Nodecl::List& env,
Analysis::NodeclSet& dep_in_set,
Analysis::NodeclSet& dep_out_set,
Analysis::NodeclSet& dep_inout_set)
{
Nodecl::NodeclBase dep_in_clause = env.find_first<Nodecl::OpenMP::DepIn>();
if (!dep_in_clause.is_null())
{
Nodecl::List dep_in_list = dep_in_clause.as<Nodecl::OpenMP::DepIn>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_in_list.begin(); it != dep_in_list.end(); ++it)
dep_in_set.insert(*it);
}
Nodecl::NodeclBase dep_out_clause = env.find_first<Nodecl::OpenMP::DepOut>();
if (!dep_out_clause.is_null())
{
Nodecl::List dep_out_list = dep_out_clause.as<Nodecl::OpenMP::DepOut>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_out_list.begin(); it != dep_out_list.end(); ++it)
dep_out_set.insert(*it);
}
Nodecl::NodeclBase dep_inout_clause = env.find_first<Nodecl::OpenMP::DepInout>();
if (!dep_inout_clause.is_null())
{
Nodecl::List dep_inout_list = dep_inout_clause.as<Nodecl::OpenMP::DepInout>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_inout_list.begin(); it != dep_inout_list.end(); ++it)
dep_inout_set.insert(*it);
}
}
static bool new_attributes_define_previous_attributes(
LintAttributes* old_attrs,
Nodecl::List new_env)
{
Analysis::NodeclSet previous_all_in;
gather_arrays_info(old_attrs->get_in(), previous_all_in, true);
gather_arrays_info(old_attrs->get_out(), previous_all_in, false);
gather_arrays_info(old_attrs->get_inout(), previous_all_in, true);
Analysis::NodeclSet dep_in_set, dep_out_set, dep_inout_set;
get_dependency_sets(new_env, dep_in_set, dep_out_set, dep_inout_set);
Analysis::NodeclSet new_all_outs = Analysis::Utils::nodecl_set_union(dep_out_set, dep_inout_set);
return check_attributes_list(previous_all_in, new_all_outs);
}
static void replace_parameters_by_arguments(
const ObjectList<Symbol>& params,
const Nodecl::List& args,
Nodecl::List& env)
{
ObjectList<Symbol>::const_iterator itp = params.begin();
Nodecl::List::const_iterator ita = args.begin();
for (; itp != params.end() && ita != args.end(); ++itp, ++ita)
{
Nodecl::Symbol itp_sym = Nodecl::Symbol::make(*itp);
Nodecl::NodeclBase real_ita = (ita->is<Nodecl::Neg>() ? ita->as<Nodecl::Neg>().get_rhs().no_conv() : *ita);
if (real_ita.is_constant())
{
for (Nodecl::List::iterator it = env.begin(); it != env.end(); ++it)
{
if (it->is<Nodecl::OpenMP::DepIn>()
|| it->is<Nodecl::OpenMP::DepOut>()
|| it->is<Nodecl::OpenMP::DepInout>())
{
Nodecl::List deps = it->as<Nodecl::OpenMP::DepIn>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator ite = deps.begin(); ite != deps.end(); )
{
if (Nodecl::Utils::structurally_equal_nodecls(Nodecl::Symbol::make(*itp), *ite))
{
ite = deps.erase(ite);
}
else
{
Nodecl::Utils::nodecl_replace_nodecl_by_structure( *ite,  itp_sym,  real_ita);
++ite;
}
}
}
}
}
for (Nodecl::List::iterator it = env.begin(); it != env.end(); ++it)
{
if (it->is<Nodecl::OpenMP::DepIn>()
|| it->is<Nodecl::OpenMP::DepOut>()
|| it->is<Nodecl::OpenMP::DepInout>())
{
Nodecl::List deps = it->as<Nodecl::OpenMP::DepIn>().get_exprs().as<Nodecl::List>();
Nodecl::Utils::nodecl_replace_nodecl_by_structure( deps,  itp_sym,  real_ita);
}
}
}
for (Nodecl::List::iterator it = env.begin(); it != env.end(); ++it)
{
if (it->is<Nodecl::OpenMP::DepIn>()
|| it->is<Nodecl::OpenMP::DepOut>()
|| it->is<Nodecl::OpenMP::DepInout>())
{
Nodecl::List deps = it->as<Nodecl::OpenMP::DepIn>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator ite = deps.begin(); ite != deps.end(); ++ite)
{
if (ite->no_conv().is<Nodecl::Dereference>())
{
if (ite->no_conv().as<Nodecl::Dereference>().get_rhs().no_conv().is<Nodecl::Reference>())
{
ite->replace(ite->no_conv().as<Nodecl::Dereference>().get_rhs().no_conv().as<Nodecl::Reference>().get_rhs());
}
}
}
}
}
}
bool OssLint::analyze_simple_node(
Analysis::Node* n,
ObjectList<Nodecl::NodeclBase>& path,
LintAttributes* attrs,
const Analysis::Utils::InductionVarList& ind_vars,
bool& continue_outer_levels)
{
Analysis::NodeclList stmts = n->get_statements();
ERROR_CONDITION(stmts.size()>1,
"Simple node with more than one statement. This is not supported for Lint\n", 0);
const Analysis::NodeclSet& undef_vars = n->get_undefined_behaviour_vars();
if (!undef_vars.empty())
{
return false;
}
if (n->is_return_node() || n->is_omp_taskwait_node())
{
return false;
}
if (stmts[0].is<Nodecl::ObjectInit>())
{
std::string sym_name = stmts[0].get_symbol().get_name();
if ((sym_name.rfind("mcc_vla_", 0) == std::string::npos)
&& (sym_name.rfind("__MERCURIUM_", 0) == std::string::npos)) {
if (!path.empty())
{
std::cerr << "lint-debug: declaration of variable '" << stmts[0].get_symbol().get_name()
<< "' in " << stmts[0].get_locus_str()
<< " prevents from propagating Lint information. "
"Consider promoting the declaration to upper levels, if possible, "
"for a better Lint optimization." << std::endl;
}
}
return false;
}
else if (stmts[0].is<Nodecl::FunctionCall>())
{
TL::Symbol enclosing_func = Nodecl::Utils::get_enclosing_function(stmts[0]);
if (stmts[0].get_symbol() == enclosing_func)
{
if (stmts[0].get_symbol().is_valid())
{   
std::cerr << "lint-debug: recursive call in " << stmts[0].get_locus_str()
<< " prevents from propagating Lint information. " << std::endl;
}
return false;
}
else
{
Nodecl::List nested_env;
bool verified = analyze_nested_call(stmts[0].as<Nodecl::FunctionCall>(), nested_env);
if (!verified)
{
return false;
}
else
{
append_to_path(path, stmts[0]);
Nodecl::FunctionCall func_call = stmts[0].as<Nodecl::FunctionCall>();
replace_parameters_by_arguments(func_call.get_called().get_symbol().get_function_parameters(),
func_call.get_arguments().as<Nodecl::List>(),
nested_env);
append_to_attrs(attrs, nested_env);
return true;
}
return false;
}
}
else if (stmts[0].is<Nodecl::Delete>()
|| stmts[0].is<Nodecl::DeleteArray>())
{
std::cerr << "lint-debug: delete statement in " << stmts[0].get_locus_str()
<< " prevents from propagating Lint information. " << std::endl;
return false;
}
bool propagate = true;
bool is_loop_condition =
n->get_outer_node()->is_for_loop()
&& n->get_children().size() == 2
&& (n->get_exit_edges().at(0)->is_true_edge()
|| n->get_exit_edges().at(0)->is_false_edge());
bool is_loop_increment =
n->get_outer_node()->is_for_loop()
&& (n->get_children().size() == 1)
&& (n->get_children().at(0)->get_children().size() == 2)
&& (n->get_children().at(0)->get_exit_edges().at(0)->is_true_edge()
|| n->get_children().at(0)->get_exit_edges().at(0)->is_false_edge());
bool is_loop_init = (n->get_children().size() == 1)
&& n->get_children().at(0)->is_for_loop()
&& n->get_children().at(0)->get_graph_related_ast().is<Nodecl::ForStatement>()
&& Nodecl::Utils::nodecl_contains_nodecl_by_pointer(
n->get_children().at(0)->get_graph_related_ast().as<Nodecl::ForStatement>().get_loop_header(),
stmts[0]);
bool path_contains_task = false;
for (ObjectList<Nodecl::NodeclBase>::iterator it = path.begin();
it != path.end() && !path_contains_task; ++it)
{
if (it->is<Nodecl::OpenMP::Task>()
|| it->is<Nodecl::OmpSs::TaskCall>())
{
path_contains_task = true;
}
}
LintAttributes* new_attrs = new LintAttributes();
bool verified = update_attributes(n, new_attrs, ind_vars);
if ((!is_loop_condition && !is_loop_increment)
|| path_contains_task)  
{
if (!verified || new_attributes_define_previous_attributes( attrs,  new_attrs))
{
if (!is_loop_condition && !is_loop_increment)
propagate = false;
continue_outer_levels = false;
}
}
if (!propagate)
{
wrap_statements(path, attrs);
}
if (verified)
{
append_to_path(path, stmts[0]);
if (!is_loop_init || path.size()>0)
{   
attrs->add_attrs(new_attrs);
}
}
return verified;
}
static void report_overprotective(
Analysis::NodeclSet overprotective_set,
std::string message,
std::string locus)
{
for (Analysis::NodeclSet:: iterator it = overprotective_set.begin(); it != overprotective_set.end(); ++it)
{
std::cerr << "lint-error: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as " << message << ", but it shouldn't." << std::endl;
}
}
static void report_missing(
Analysis::NodeclSet missing_set,
std::string message,
std::string locus)
{
for (Analysis::NodeclSet:: iterator it = missing_set.begin(); it != missing_set.end(); ++it)
{
std::cerr << "lint-error: task in "<< locus
<< " does not define variable " << it->prettyprint()
<< " as " << message << ", but it should." << std::endl;
}
}
static void report_mistaken(
std::multimap<Nodecl::NodeclBase, Nodecl::NodeclBase,
Nodecl::Utils::Nodecl_structural_less> mistaken_map,
std::string message,
std::string locus)
{
for (std::multimap<Nodecl::NodeclBase, Nodecl::NodeclBase,
Nodecl::Utils::Nodecl_structural_less>::iterator it = mistaken_map.begin();
it != mistaken_map.end(); ++it)
{
std::cerr << "lint-error: task in " << locus
<< " defines an array " << it->first.prettyprint()
<< " as " << message
<< ", but the range should be " << it->second.prettyprint()
<< "." << std::endl;
}
}
static bool set_contains_container_object(
const Nodecl::NodeclBase& v,
Analysis::NodeclSet& set)
{
bool match = false;
Nodecl::NodeclBase v_cp;
for (Analysis::NodeclSet::iterator ita = set.begin();
ita != set.end() && !match; ++ita)
{
v_cp = v.shallow_copy();
while (!v_cp.is_null())
{
if (Nodecl::Utils::structurally_equal_nodecls(v_cp, *ita, true))
{
match = true;
break;
}
else if (v_cp.is<Nodecl::ArraySubscript>())
{
v_cp = v_cp.as<Nodecl::ArraySubscript>().get_subscripted().no_conv();
}
else if (v_cp.is<Nodecl::ClassMemberAccess>())
{
v_cp = v_cp.as<Nodecl::ClassMemberAccess>().get_lhs().no_conv();
}
else if (v_cp.is<Nodecl::Dereference>())
{
v_cp = v_cp.as<Nodecl::Dereference>().get_rhs().no_conv();
}
else
{
v_cp = Nodecl::NodeclBase::null(); 
break;
}
}
}
return match;
}
static ObjectList<Nodecl::NodeclBase> dependency_comparison(
Nodecl::NodeclBase itu_cp,
Analysis::NodeclSet& analyzed_set)
{
ObjectList<Nodecl::NodeclBase> found;
bool exact_match = false;
for (Analysis::NodeclSet::iterator ita = analyzed_set.begin();
ita != analyzed_set.end() && !exact_match; ++ita)
{
Nodecl::NodeclBase v_a = ita->no_conv().shallow_copy();
while (!v_a.is_null())
{
if (Nodecl::Utils::structurally_equal_nodecls(itu_cp.no_conv(), v_a,
true))
{
exact_match = Nodecl::Utils::structurally_equal_nodecls(*ita, v_a, true);
found.append(*ita);
break;
}
else if (v_a.is<Nodecl::ArraySubscript>())
{
if (itu_cp.is<Nodecl::ArraySubscript>())
{
Nodecl::NodeclBase different_subscript;
if (!Nodecl::Utils::structurally_equal_nodecls(itu_cp.as<Nodecl::ArraySubscript>().get_subscripted().no_conv(),
v_a.as<Nodecl::ArraySubscript>().get_subscripted().no_conv(),
true))
{
v_a = Nodecl::NodeclBase::null(); 
break;
}
Nodecl::List itu_subscripts = itu_cp.as<Nodecl::ArraySubscript>().get_subscripts().as<Nodecl::List>();
Nodecl::List ita_subscripts = v_a.as<Nodecl::ArraySubscript>().get_subscripts().as<Nodecl::List>();
Nodecl::List::iterator itus = itu_subscripts.begin();
Nodecl::List::iterator itas = ita_subscripts.begin();
for (; itus != itu_subscripts.end() && itas != itu_subscripts.end(); ++itus, ++itas)
{
if (itus->no_conv().is<Nodecl::Range>() && itas->no_conv().is<Nodecl::Range>())
{
Scope sc = itus->retrieve_context();
Nodecl::NodeclBase itus_lb = itus->no_conv().as<Nodecl::Range>().get_lower().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itus_lb,  false);
Nodecl::NodeclBase itas_lb = itas->no_conv().as<Nodecl::Range>().get_lower().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itas_lb,  false);
Nodecl::Minus lb_sub = Nodecl::Minus::make(itus_lb, itas_lb, itus_lb.get_type());
TL::Optimizations::canonicalize_and_fold(lb_sub,  false);
if (!Nodecl::Utils::structurally_equal_nodecls(itus_lb, itas_lb,  true)
&& (!lb_sub.is_constant() || !const_value_is_zero(lb_sub.get_constant())))
{
different_subscript = v_a;
continue;
}
Nodecl::NodeclBase itus_ub = itus->no_conv().as<Nodecl::Range>().get_upper().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itus_ub,  false);
Nodecl::NodeclBase itas_ub = itas->no_conv().as<Nodecl::Range>().get_upper().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itas_ub,  false);
Nodecl::NodeclBase ub_sub = Nodecl::Minus::make(itus_ub, itas_ub, itus_ub.get_type());
TL::Optimizations::canonicalize_and_fold(ub_sub,  false);
if (!Nodecl::Utils::structurally_equal_nodecls(itus_ub, itas_ub,  true)
&& (!ub_sub.is_constant() || !const_value_is_zero(ub_sub.get_constant())))
{
different_subscript = v_a;
continue;
}
}
else
{
Nodecl::NodeclBase canon_itus = itus->no_conv().shallow_copy();
TL::Optimizations::canonicalize_and_fold(canon_itus,  false);
Nodecl::NodeclBase canon_itas = itas->no_conv().shallow_copy();
TL::Optimizations::canonicalize_and_fold(canon_itas,  false);
if (!Nodecl::Utils::structurally_equal_nodecls(canon_itus, canon_itas,
true))
{
different_subscript = v_a;
continue;
}
}
}
if (different_subscript.is_null()
&& itu_subscripts.size() <= ita_subscripts.size()) 
{   
exact_match = Nodecl::Utils::structurally_equal_nodecls(*ita, v_a, true);
found.append(*ita);
break;
}
else
{
v_a = Nodecl::NodeclBase::null(); 
break;
}
}
else
{
v_a = v_a.as<Nodecl::ArraySubscript>().get_subscripted().no_conv();
}
}
else if (v_a.is<Nodecl::ClassMemberAccess>())
{
v_a = v_a.as<Nodecl::ClassMemberAccess>().get_lhs().no_conv();
}
else if (itu_cp.is<Nodecl::Dereference>()
&& Nodecl::Utils::structurally_equal_nodecls(itu_cp.as<Nodecl::Dereference>().get_rhs().no_conv(),
*ita, true))
{
found.append(*ita);
break;
}
else
{
v_a = Nodecl::NodeclBase::null(); 
break;
}
}
}
return found;
}
static bool check_lint_task_dependencies(
Analysis::NodeclSet user_in_set,
Analysis::NodeclSet& analyzed_in_set,
Analysis::NodeclSet user_out_set,
Analysis::NodeclSet& analyzed_out_set,
Analysis::NodeclSet user_inout_set,
Analysis::NodeclSet& analyzed_inout_set,
std::string locus)
{
bool error = false;
Analysis::NodeclSet reported;
Analysis::NodeclSet in_match;
for (Analysis::NodeclSet::iterator it = user_in_set.begin(); it != user_in_set.end(); ++it)
{
Nodecl::NodeclBase v = it->no_conv().shallow_copy();
TL::Optimizations::canonicalize_and_fold(v,  false);
ObjectList<Nodecl::NodeclBase> analysis_v = dependency_comparison(v, analyzed_in_set);
if (analyzed_in_set.find(v) != analyzed_in_set.end()
|| !analysis_v.empty())
{
in_match.insert(analysis_v.begin(), analysis_v.end());
}
else
{
reported.insert(v);
if (analyzed_out_set.find(v) != analyzed_out_set.end()
|| !dependency_comparison(v, analyzed_out_set).empty())
{
error = true;
std::cerr << "lint-error: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as in, but it should be out." << std::endl;
}
else if (analyzed_inout_set.find(v) != analyzed_inout_set.end()
|| !dependency_comparison(v, analyzed_inout_set).empty())
{
error = true;
std::cerr << "lint-error: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as in, but it should be inout." << std::endl;
}
else
{
std::cerr << "lint-warning: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as in, but it has no dependency on this variable." << std::endl;
}
}
}
Analysis::NodeclSet out_match;
for (Analysis::NodeclSet::iterator it = user_out_set.begin(); it != user_out_set.end(); ++it)
{
Nodecl::NodeclBase v = it->no_conv().shallow_copy();
TL::Optimizations::canonicalize_and_fold(v,  false);
ObjectList<Nodecl::NodeclBase> analysis_v = dependency_comparison(v, analyzed_out_set);
if (analyzed_out_set.find(v) != analyzed_out_set.end()
|| !analysis_v.empty())
{
out_match.insert(analysis_v.begin(), analysis_v.end());
}
else
{
reported.insert(v);
if (analyzed_in_set.find(v) != analyzed_in_set.end()
|| !dependency_comparison(v, analyzed_in_set).empty())
{
error = true;
std::cerr << "lint-error: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as out, but it should be in." << std::endl;
}
else if (analyzed_inout_set.find(v) != analyzed_inout_set.end()
|| !dependency_comparison(*it, analyzed_inout_set).empty())
{
error = true;
std::cerr << "lint-error: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as out, but it should be inout." << std::endl;
}
else
{
std::cerr << "lint-warning: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as out, but it has no dependency on this variable." << std::endl;
}
}
}
Analysis::NodeclSet inout_match;
for (Analysis::NodeclSet::iterator it = user_inout_set.begin(); it != user_inout_set.end(); ++it)
{
Nodecl::NodeclBase v = it->no_conv().shallow_copy();
TL::Optimizations::canonicalize_and_fold(v,  false);
ObjectList<Nodecl::NodeclBase> analysis_v = dependency_comparison(v, analyzed_inout_set);
if (analyzed_inout_set.find(v) != analyzed_inout_set.end()
|| !analysis_v.empty())
{
inout_match.insert(analysis_v.begin(), analysis_v.end());
}
else
{
reported.insert(v);
if (analyzed_in_set.find(v) != analyzed_in_set.end()
|| !dependency_comparison(v, analyzed_in_set).empty())
{
std::cerr << "lint-warning: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as inout, but it should be in." << std::endl;
}
else if (analyzed_out_set.find(v) != analyzed_out_set.end()
|| !dependency_comparison(v, analyzed_out_set).empty())
{
std::cerr << "lint-warning: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as inout, but it should be out." << std::endl;
}
else
{
std::cerr << "lint-warning: task in "<< locus
<< " defines variable " << it->prettyprint()
<< " as inout, but it has no dependency on this variable." << std::endl;
}
}
}
Analysis::NodeclSet in_missing = Analysis::Utils::nodecl_set_difference(analyzed_in_set, in_match);
Analysis::NodeclSet unreported_in = Analysis::Utils::nodecl_set_difference(in_missing, reported);
for (Analysis::NodeclSet::iterator it = unreported_in.begin(); it != unreported_in.end(); ++it)
{
if (!set_contains_container_object(*it, user_inout_set))
{
error = true;
std::cerr << "lint-error: task in "<< locus
<< " does not define variable " << it->prettyprint()
<< " as in, but it should." << std::endl;
}
}
Analysis::NodeclSet out_missing = Analysis::Utils::nodecl_set_difference(analyzed_out_set, out_match);
Analysis::NodeclSet unreported_out = Analysis::Utils::nodecl_set_difference(out_missing, reported);
for (Analysis::NodeclSet::iterator it = unreported_out.begin(); it != unreported_out.end(); ++it)
{
error = true;
std::cerr << "lint-error: task in "<< locus
<< " does not define variable " << it->prettyprint()
<< " as out, but it should." << std::endl;
}
Analysis::NodeclSet inout_missing = Analysis::Utils::nodecl_set_difference(analyzed_inout_set, inout_match);
Analysis::NodeclSet unreported_inout = Analysis::Utils::nodecl_set_difference(inout_missing, reported);
for (Analysis::NodeclSet::iterator it = unreported_inout.begin(); it != unreported_inout.end(); ++it)
{
error = true;
std::cerr << "lint-error: task in "<< locus
<< " does not define variable " << it->prettyprint()
<< " as inout, but it should." << std::endl;
}
return error;
}
static void check_lint_task_memory(
Analysis::NodeclSet user_set,
Analysis::NodeclSet& analyzed_set,
Analysis::NodeclSet& overprotective_set,
Analysis::NodeclSet& missing_set,
std::multimap<Nodecl::NodeclBase, Nodecl::NodeclBase,
Nodecl::Utils::Nodecl_structural_less>& mistaken_map)
{
Analysis::NodeclSet user_analysis_match;
for (Analysis::NodeclSet::iterator itu = user_set.begin(); itu != user_set.end(); ++itu)
{
Nodecl::NodeclBase itu_cp = Nodecl::Utils::deep_copy(*itu, itu->retrieve_context());
if (analyzed_set.find(itu_cp) != analyzed_set.end())
{
user_analysis_match.insert(itu_cp);
continue;
}
Analysis::NodeclSet::iterator ita = analyzed_set.begin();
Nodecl::NodeclBase different_subscript;
for ( ; ita != analyzed_set.end(); ++ita)
{
if (itu_cp.get_kind() != ita->get_kind())
continue;
if (itu_cp.is<Nodecl::ArraySubscript>())
{
if (!Nodecl::Utils::structurally_equal_nodecls(itu_cp.as<Nodecl::ArraySubscript>().get_subscripted().no_conv(),
ita->as<Nodecl::ArraySubscript>().get_subscripted().no_conv(),
true))
continue;
Nodecl::List itu_subscripts = itu_cp.as<Nodecl::ArraySubscript>().get_subscripts().as<Nodecl::List>();
Nodecl::List ita_subscripts = ita->as<Nodecl::ArraySubscript>().get_subscripts().as<Nodecl::List>();
if (itu_subscripts.size() != ita_subscripts.size())
continue;
Nodecl::List::iterator itus = itu_subscripts.begin();
Nodecl::List::iterator itas = ita_subscripts.begin();
for (; itus != itu_subscripts.end(); ++itus, ++itas)
{
if (itus->no_conv().is<Nodecl::Range>() && itas->no_conv().is<Nodecl::Range>())
{
Scope sc = itus->retrieve_context();
Nodecl::NodeclBase itus_lb = itus->no_conv().as<Nodecl::Range>().get_lower().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itus_lb,  false);
Nodecl::NodeclBase itas_lb = itas->no_conv().as<Nodecl::Range>().get_lower().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itas_lb,  false);
Nodecl::Minus lb_sub = Nodecl::Minus::make(itus_lb, itas_lb, itus_lb.get_type());
TL::Optimizations::canonicalize_and_fold(lb_sub,  false);
if (!Nodecl::Utils::structurally_equal_nodecls(itus_lb, itas_lb,  true)
&& (!lb_sub.is_constant() || !const_value_is_zero(lb_sub.get_constant())))
{
different_subscript = *ita;
continue;
}
Nodecl::NodeclBase itus_ub = itus->no_conv().as<Nodecl::Range>().get_upper().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itus_ub,  false);
Nodecl::NodeclBase itas_ub = itas->no_conv().as<Nodecl::Range>().get_upper().shallow_copy();
TL::Optimizations::canonicalize_and_fold(itas_ub,  false);
Nodecl::NodeclBase ub_sub = Nodecl::Minus::make(itus_ub, itas_ub, itus_ub.get_type());
TL::Optimizations::canonicalize_and_fold(ub_sub,  false);
if (!Nodecl::Utils::structurally_equal_nodecls(itus_ub, itas_ub,  true)
&& (!ub_sub.is_constant() || !const_value_is_zero(ub_sub.get_constant())))
{
different_subscript = *ita;
continue;
}
}
else
{
Nodecl::NodeclBase canon_itus = itus->no_conv().shallow_copy();
TL::Optimizations::canonicalize_and_fold(canon_itus,  false);
Nodecl::NodeclBase canon_itas = itas->no_conv().shallow_copy();
TL::Optimizations::canonicalize_and_fold(canon_itas,  false);
if (!Nodecl::Utils::structurally_equal_nodecls(canon_itus, canon_itas,
true))
{
different_subscript = *ita;
continue;
}
}
}
if (different_subscript.is_null())
break; 
}
else
{
if (Nodecl::Utils::structurally_equal_nodecls(itu_cp.no_conv(), ita->no_conv(),
true))
break;
}
}
if (ita == analyzed_set.end())
{
if (!different_subscript.is_null()){
mistaken_map.insert(std::pair<Nodecl::NodeclBase, Nodecl::NodeclBase>(itu_cp, different_subscript));
} else {
overprotective_set.insert(itu_cp);
}
}
else
{
user_analysis_match.insert(*ita);
}
}
missing_set = Analysis::Utils::nodecl_set_difference(analyzed_set, user_analysis_match);
for (std::multimap<Nodecl::NodeclBase, Nodecl::NodeclBase, Nodecl::Utils::Nodecl_structural_less>::iterator
it = mistaken_map.begin(); it != mistaken_map.end(); ++it)
{
Analysis::NodeclSet::iterator itm = missing_set.find(it->second);
if (itm != missing_set.end())
missing_set.erase(itm);
}
}
Analysis::ExtensibleGraph* OssLint::get_pcfg_by_symbol(const TL::Symbol& s)
{
for (ObjectList<Analysis::ExtensibleGraph*>::const_iterator it = _pcfgs.begin(); it != _pcfgs.end(); ++it)
{
Symbol pcfg_func_sym = (*it)->get_function_symbol();
if (pcfg_func_sym.is_valid()
&& pcfg_func_sym == s)
{
return *it;
}
}
return NULL;
}
static void add_to_environment(const Nodecl::List& source, Nodecl::List& target)
{
for (Nodecl::List::iterator its = source.begin(); its != source.end(); its++)
{
if (its->is<Nodecl::OpenMP::DepIn>())
{
Nodecl::List new_in = its->as<Nodecl::OpenMP::DepIn>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator iti = new_in.begin(); iti != new_in.end(); ++iti)
{
Nodecl::OpenMP::DepOut target_out = target.find_first<Nodecl::OpenMP::DepOut>();
Nodecl::OpenMP::DepInout target_inout = target.find_first<Nodecl::OpenMP::DepInout>();
Nodecl::NodeclBase canon_iti = iti->shallow_copy();
TL::Optimizations::canonicalize_and_fold(canon_iti,  false);
if ((target_out.is_null()
|| !Nodecl::Utils::nodecl_contains_nodecl_by_structure(target_out, canon_iti))
&& (target_inout.is_null()
|| !Nodecl::Utils::nodecl_contains_nodecl_by_structure(target_inout, canon_iti)))
{
bool found_in_set = false;
if (!target_inout.is_null())
{
Nodecl::List target_inout_exprs = target_inout.get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator ite = target_inout_exprs.begin();
ite != target_inout_exprs.end(); ++ite)
{
bool found_in_set_iter = false;
Nodecl::NodeclBase tmp = canon_iti.shallow_copy();
while (!found_in_set_iter && !tmp.is_null())
{
if (Nodecl::Utils::structurally_equal_nodecls(tmp, *ite, true))
{
found_in_set = found_in_set_iter = true;
Nodecl::Utils::remove_from_enclosing_list(*ite);
if (target_inout.is_null())
{
Nodecl::NodeclBase new_target_inout = Nodecl::OpenMP::DepInout::make(Nodecl::List::make(canon_iti));
target.append(new_target_inout);
}
else
{
target_inout.get_exprs().as<Nodecl::List>().append(canon_iti);
}
break;
}
else if (tmp.is<Nodecl::ArraySubscript>())
{
tmp = tmp.as<Nodecl::ArraySubscript>().get_subscripted();
}
else if (tmp.is<Nodecl::ClassMemberAccess>())
{
tmp = tmp.as<Nodecl::ClassMemberAccess>().get_lhs();
}
else if (tmp.is<Nodecl::Conversion>())
{
tmp = tmp.as<Nodecl::Conversion>().get_nest();
}
else
{
tmp = Nodecl::NodeclBase::null();
}
}
}
}
if (!found_in_set)
{   
Nodecl::OpenMP::DepIn target_in = target.find_first<Nodecl::OpenMP::DepIn>();
if (target_in.is_null())
{
Nodecl::NodeclBase new_target_in = Nodecl::OpenMP::DepIn::make(Nodecl::List::make(canon_iti));
target.append(new_target_in);
}
else
{
target_in.get_exprs().as<Nodecl::List>().append(canon_iti);
}
}
}
}
}
else if (its->is<Nodecl::OpenMP::DepOut>())
{
Nodecl::List new_out = its->as<Nodecl::OpenMP::DepOut>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator iti = new_out.begin(); iti != new_out.end(); ++iti)
{
Nodecl::OpenMP::DepInout target_inout = target.find_first<Nodecl::OpenMP::DepInout>();
Nodecl::NodeclBase canon_iti = iti->shallow_copy();
TL::Optimizations::canonicalize_and_fold(canon_iti,  false);
if (target_inout.is_null()
|| !Nodecl::Utils::nodecl_contains_nodecl_by_structure(target_inout, canon_iti))
{
Nodecl::OpenMP::DepIn target_in = target.find_first<Nodecl::OpenMP::DepIn>();
if (!target_in.is_null())
{
bool found_in_set = false;
Nodecl::List target_in_exprs = target_in.get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator ite = target_in_exprs.begin();
ite != target_in_exprs.end(); ++ite)
{
bool found_in_set_iter = false;
Nodecl::NodeclBase tmp = ite->shallow_copy();
while (!found_in_set_iter && !tmp.is_null())
{
if (Nodecl::Utils::structurally_equal_nodecls(tmp, canon_iti, true))
{
found_in_set = found_in_set_iter = true;
Nodecl::Utils::remove_from_enclosing_list(*ite);
if (target_inout.is_null())
{
Nodecl::NodeclBase new_target_inout = Nodecl::OpenMP::DepInout::make(Nodecl::List::make(canon_iti));
target.append(new_target_inout);
}
else
{
target_inout.get_exprs().as<Nodecl::List>().append(canon_iti);
}
break;
}
else if (tmp.is<Nodecl::ArraySubscript>())
{
tmp = tmp.as<Nodecl::ArraySubscript>().get_subscripted();
}
else if (tmp.is<Nodecl::ClassMemberAccess>())
{
tmp = tmp.as<Nodecl::ClassMemberAccess>().get_lhs();
}
else if (tmp.is<Nodecl::Conversion>())
{
tmp = tmp.as<Nodecl::Conversion>().get_nest();
}
else
{
tmp = Nodecl::NodeclBase::null();
}
}
}
if (!found_in_set)
{
for (Nodecl::List::iterator ite = target_in_exprs.begin();
ite != target_in_exprs.end(); ++ite)
{
bool found_in_set_iter = false;
Nodecl::NodeclBase tmp = canon_iti.shallow_copy();
while (!found_in_set_iter && !tmp.is_null())
{
if (Nodecl::Utils::structurally_equal_nodecls(*ite, tmp, true))
{
found_in_set = found_in_set_iter = true;
Nodecl::Utils::remove_from_enclosing_list(*ite);
if (target_inout.is_null())
{
Nodecl::NodeclBase new_target_inout = Nodecl::OpenMP::DepInout::make(Nodecl::List::make(canon_iti));
target.append(new_target_inout);
}
else
{
target_inout.get_exprs().as<Nodecl::List>().append(canon_iti);
}
break;
}
else if (tmp.is<Nodecl::ArraySubscript>())
{
tmp = tmp.as<Nodecl::ArraySubscript>().get_subscripted();
}
else if (tmp.is<Nodecl::ClassMemberAccess>())
{
tmp = tmp.as<Nodecl::ClassMemberAccess>().get_lhs();
}
else if (tmp.is<Nodecl::Conversion>())
{
tmp = tmp.as<Nodecl::Conversion>().get_nest();
}
else
{
tmp = Nodecl::NodeclBase::null();
}
}
}
if (!found_in_set)
{
Nodecl::OpenMP::DepOut target_out = target.find_first<Nodecl::OpenMP::DepOut>();
if (target_out.is_null())
{
Nodecl::NodeclBase new_target_out = Nodecl::OpenMP::DepOut::make(Nodecl::List::make(canon_iti));
target.append(new_target_out);
}
else
{
target_out.get_exprs().as<Nodecl::List>().append(canon_iti);
}
}
}
}
}
}
}
else if (its->is<Nodecl::OpenMP::DepInout>())
{
Nodecl::List new_inout = its->as<Nodecl::OpenMP::DepInout>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator iti = new_inout.begin(); iti != new_inout.end(); ++iti)
{
Nodecl::OpenMP::DepOut target_out = target.find_first<Nodecl::OpenMP::DepOut>();
Nodecl::NodeclBase canon_iti = iti->shallow_copy();
TL::Optimizations::canonicalize_and_fold(canon_iti,  false);
if (target_out.is_null()
|| !Nodecl::Utils::nodecl_contains_nodecl_by_structure(target_out, canon_iti))
{
Nodecl::OpenMP::DepIn target_in = target.find_first<Nodecl::OpenMP::DepIn>();
if (!target_in.is_null())
{
bool found_in_set = false;
Nodecl::List target_in_exprs = target_in.get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator ite = target_in_exprs.begin();
ite != target_in_exprs.end(); ++ite)
{
bool found_in_set_iter = false;
Nodecl::NodeclBase tmp = canon_iti.shallow_copy();
while (!found_in_set_iter && !tmp.is_null())
{
if (Nodecl::Utils::structurally_equal_nodecls(*ite, tmp, true))
{
found_in_set = found_in_set_iter = true;
Nodecl::Utils::remove_from_enclosing_list(*ite);
Nodecl::OpenMP::DepInout target_inout = target.find_first<Nodecl::OpenMP::DepInout>();
if (target_inout.is_null())
{
Nodecl::NodeclBase new_target_inout = Nodecl::OpenMP::DepInout::make(Nodecl::List::make(canon_iti));
target.append(new_target_inout);
}
else
{
target_inout.get_exprs().as<Nodecl::List>().append(canon_iti);
}
break;
}
else if (tmp.is<Nodecl::ArraySubscript>())
{
tmp = tmp.as<Nodecl::ArraySubscript>().get_subscripted();
}
else if (tmp.is<Nodecl::ClassMemberAccess>())
{
tmp = tmp.as<Nodecl::ClassMemberAccess>().get_lhs();
}
else if (tmp.is<Nodecl::Conversion>())
{
tmp = tmp.as<Nodecl::Conversion>().get_nest();
}
else
{
tmp = Nodecl::NodeclBase::null();
}
}
}
if (!found_in_set)
{
for (Nodecl::List::iterator ite = target_in_exprs.begin();
ite != target_in_exprs.end(); ++ite)
{
bool found_in_set_iter = false;
Nodecl::NodeclBase tmp = ite->shallow_copy();
while (!found_in_set_iter && !tmp.is_null())
{
if (Nodecl::Utils::structurally_equal_nodecls(tmp, canon_iti, true))
{
found_in_set = found_in_set_iter = true;
Nodecl::Utils::remove_from_enclosing_list(*ite);
Nodecl::OpenMP::DepInout target_inout = target.find_first<Nodecl::OpenMP::DepInout>();
if (target_inout.is_null())
{
Nodecl::NodeclBase new_target_inout = Nodecl::OpenMP::DepInout::make(Nodecl::List::make(canon_iti));
target.append(new_target_inout);
}
else
{
target_inout.get_exprs().as<Nodecl::List>().append(canon_iti);
}
break;
}
else if (tmp.is<Nodecl::ArraySubscript>())
{
tmp = tmp.as<Nodecl::ArraySubscript>().get_subscripted();
}
else if (tmp.is<Nodecl::ClassMemberAccess>())
{
tmp = tmp.as<Nodecl::ClassMemberAccess>().get_lhs();
}
else if (tmp.is<Nodecl::Conversion>())
{
tmp = tmp.as<Nodecl::Conversion>().get_nest();
}
else
{
tmp = Nodecl::NodeclBase::null();
}
}
}
if (!found_in_set)
{   
Nodecl::OpenMP::DepInout target_inout = target.find_first<Nodecl::OpenMP::DepInout>();
if (target_inout.is_null())
{
Nodecl::NodeclBase new_target_inout = Nodecl::OpenMP::DepInout::make(Nodecl::List::make(canon_iti));
target.append(new_target_inout);
}
else
{
target_inout.get_exprs().as<Nodecl::List>().append(canon_iti);
}
}
}
}
}
}
}
else if (its->is<Nodecl::OmpSs::LintAlloc>())
{
Nodecl::OmpSs::LintAlloc target_alloc = target.find_first<Nodecl::OmpSs::LintAlloc>();
if (target_alloc.is_null())
{
target.append(its->shallow_copy());
}
else
{
Nodecl::List new_alloc = its->as<Nodecl::OmpSs::LintAlloc>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator iti = new_alloc.begin(); iti != new_alloc.end(); ++iti)
{
target_alloc.get_exprs().as<Nodecl::List>().append(iti->shallow_copy());
}
}
}
else if (its->is<Nodecl::OmpSs::LintFree>())
{
Nodecl::OmpSs::LintFree target_free = target.find_first<Nodecl::OmpSs::LintFree>();
if (target_free.is_null())
{
target.append(its->shallow_copy());
}
else
{
Nodecl::List new_free = its->as<Nodecl::OmpSs::LintFree>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator iti = new_free.begin(); iti != new_free.end(); ++iti)
{
target_free.get_exprs().as<Nodecl::List>().append(iti->shallow_copy());
}
}
}
else
{}  
}
}
static void remove_from_list(Nodecl::List& env,
Nodecl::List& exprs,
const ObjectList<Symbol>& private_syms)
{
Nodecl::NodeclBase dep_clause = Nodecl::Utils::get_enclosing_list(exprs).get_parent();
for (Nodecl::List::iterator it = exprs.begin(); it != exprs.end(); )
{
DataReference it_dr(*it);
Symbol base_sym = it_dr.get_base_symbol();
if (private_syms.contains(base_sym))
{
it = exprs.erase(it);
}
else
{
++it;
}
}
if (exprs.empty())
{   
if (dep_clause.is<Nodecl::OpenMP::DepIn>())
{
for (Nodecl::List::iterator it = env.begin(); it != env.end(); ++it)
{
if (it->is<Nodecl::OpenMP::DepIn>())
{
Nodecl::Utils::remove_from_enclosing_list(*it);
break;
}
}
}
else if (dep_clause.is<Nodecl::OpenMP::DepOut>())
{
for (Nodecl::List::iterator it = env.begin(); it != env.end(); ++it)
{
if (it->is<Nodecl::OpenMP::DepOut>())
{
Nodecl::Utils::remove_from_enclosing_list(*it);
break;
}
}
}
else if (dep_clause.is<Nodecl::OpenMP::DepInout>())
{
for (Nodecl::List::iterator it = env.begin(); it != env.end(); ++it)
{
if (it->is<Nodecl::OpenMP::DepInout>())
{
Nodecl::Utils::remove_from_enclosing_list(*it);
break;
}
}
}
}
}
static void remove_function_private_variables(
Nodecl::List& env, const ObjectList<Symbol>& private_syms)
{
Nodecl::OpenMP::DepIn in_clause = env.find_first<Nodecl::OpenMP::DepIn>();
if (!in_clause.is_null())
{
Nodecl::List dep_in = in_clause.get_exprs().as<Nodecl::List>();
remove_from_list(env, dep_in, private_syms);
}
Nodecl::OpenMP::DepOut out_clause = env.find_first<Nodecl::OpenMP::DepOut>();
if (!out_clause.is_null())
{
Nodecl::List dep_out = out_clause.get_exprs().as<Nodecl::List>();
remove_from_list(env, dep_out, private_syms);
}
Nodecl::OpenMP::DepInout inout_clause = env.find_first<Nodecl::OpenMP::DepInout>();
if (!inout_clause.is_null())
{
Nodecl::List dep_inout = inout_clause.get_exprs().as<Nodecl::List>();
remove_from_list(env, dep_inout, private_syms);
}
}
bool OssLint::analyze_nested_call(
const Nodecl::FunctionCall& function_call,
Nodecl::List& nested_env)
{
TL::Symbol called_sym = function_call.get_called().get_symbol();
if (Nodecl::Utils::get_enclosing_function(function_call) == called_sym)
{   
std::cerr << "lint-debug: recursive call in " << function_call.get_locus_str()
<< " prevents verification. " << std::endl;
return false;
}
bool only_tasks_tmp = _only_tasks;
_only_tasks = false;
Analysis::ExtensibleGraph* pcfg = get_pcfg_by_symbol(called_sym);
if (pcfg == NULL)
{
std::cerr << "lint-debug: call to function with non-accessible code "
<< called_sym.get_name() << ". This prevents verification." << std::endl;
_only_tasks = only_tasks_tmp;
return false;
}
if (analyzed_pcfgs.find(pcfg) == analyzed_pcfgs.end())
{   
oss_lint_analyze_all(pcfg);
}
_only_tasks = only_tasks_tmp;
Nodecl::NodeclBase ast = pcfg->get_graph()->get_graph_related_ast();
Nodecl::List func_stmts = ast.as<Nodecl::FunctionCode>().get_statements().as<Nodecl::List>();
Nodecl::List cmp_stmts = func_stmts.at(0).as<Nodecl::CompoundStatement>().get_statements().as<Nodecl::List>();
bool verified = true;
for (Nodecl::List::iterator it = cmp_stmts.begin(); it != cmp_stmts.end() && verified; ++it)
{
if (!it->is<Nodecl::CxxDef>()
&& !it->is<Nodecl::ObjectInit>()
&& !it->is<Nodecl::OmpSs::Lint>()
&& (!it->is<Nodecl::OpenMP::Task>()
|| !it->is<Nodecl::OmpSs::TaskCall>()
|| it->as<Nodecl::OpenMP::Task>().get_environment().as<Nodecl::List>().find_first<Nodecl::OmpSs::LintVerified>().is_null())
&& (!it->is<Nodecl::Context>()
|| it->as<Nodecl::Context>().get_in_context().as<Nodecl::List>().size() != 1
|| !it->as<Nodecl::Context>().get_in_context().as<Nodecl::List>().at(0).is<Nodecl::OmpSs::Lint>())
&& !it->is<Nodecl::ReturnStatement>())   
{
verified = false;
}
}
if (!verified)
return false;
ObjectList<Symbol> func_syms =
func_stmts.at(0).as<Nodecl::CompoundStatement>().retrieve_context().get_all_symbols(false);
ObjectList<Symbol> func_params = called_sym.get_function_parameters();
ObjectList<Symbol> private_syms;
for (ObjectList<Symbol>::iterator it = func_syms.begin(); it != func_syms.end(); ++it)
{
if (!func_params.contains(*it))
private_syms.insert(*it);
}
for (Nodecl::List::iterator it = cmp_stmts.begin(); it != cmp_stmts.end(); ++it)
{
if (it->is<Nodecl::CxxDef>() || it->is<Nodecl::ObjectInit>() || it->is<Nodecl::ReturnStatement>())
continue;
if (it->is<Nodecl::OmpSs::Lint>())
{
Nodecl::List env = it->as<Nodecl::OmpSs::Lint>().get_environment().as<Nodecl::List>();
remove_function_private_variables(env, private_syms);
add_to_environment( it->as<Nodecl::OmpSs::Lint>().get_environment().as<Nodecl::List>(),
nested_env);
}
else if (it->as<Nodecl::Context>().get_in_context().as<Nodecl::List>().at(0).is<Nodecl::OmpSs::Lint>())
{
Nodecl::List env =
it->as<Nodecl::Context>().get_in_context().as<Nodecl::List>().at(0).as<Nodecl::OmpSs::Lint>().get_environment().as<Nodecl::List>();
remove_function_private_variables(env, private_syms);
add_to_environment( it->as<Nodecl::Context>().get_in_context().as<Nodecl::List>().at(0).as<Nodecl::OmpSs::Lint>().get_environment().as<Nodecl::List>(),
nested_env);
}
else if (it->is<Nodecl::OpenMP::Task>() || it->is<Nodecl::OmpSs::TaskCall>())
{   
Nodecl::List env = it->as<Nodecl::OpenMP::Task>().get_environment().as<Nodecl::List>();
add_to_environment( env,  nested_env);
}
else
{
internal_error("Unreachable code.\n", 0);
}
}
return true;
}
static void purge_firstprivate(
const Nodecl::NodeclBase& fp_clause,
Analysis::NodeclSet& analyzed_in_set,
Analysis::NodeclSet& analyzed_out_set,
Analysis::NodeclSet& analyzed_inout_set)
{
if (fp_clause.is_null())
return;
Nodecl::List fp_list = fp_clause.as<Nodecl::OpenMP::Firstprivate>().get_symbols().as<Nodecl::List>();
ObjectList<Nodecl::NodeclBase> fp_obj_list = fp_list.to_object_list();
for (Analysis::NodeclSet::iterator it = analyzed_in_set.begin(); it != analyzed_in_set.end(); )
{
if (Nodecl::Utils::list_contains_nodecl_by_structure( fp_obj_list,  *it))
analyzed_in_set.erase(it++);
else
++it;
}
for (Analysis::NodeclSet::iterator it = analyzed_out_set.begin(); it != analyzed_out_set.end(); )
{
if (Nodecl::Utils::list_contains_nodecl_by_structure( fp_obj_list,  *it))
analyzed_out_set.erase(it++);
else
++it;
}
for (Analysis::NodeclSet::iterator it = analyzed_inout_set.begin(); it != analyzed_inout_set.end(); )
{
if (Nodecl::Utils::list_contains_nodecl_by_structure( fp_obj_list,  *it))
analyzed_inout_set.erase(it++);
else
++it;
}
}
bool OssLint::analyze_task_node(
Analysis::Node* n,
ObjectList<Nodecl::NodeclBase>& path,
LintAttributes* attrs,
Analysis::Utils::InductionVarList& ind_vars,
bool& continue_outer_levels)
{
Nodecl::NodeclBase ast = n->get_graph_related_ast();
Nodecl::List env = n->get_graph_related_ast().as<Nodecl::OpenMP::Task>().get_environment().as<Nodecl::List>();
if (n->has_verified_clause())
{
if (new_attributes_define_previous_attributes( attrs,  env))
{
wrap_statements(path, attrs);
}
append_to_path(path, ast);
append_to_attrs(attrs, env);
return true;
}
bool is_task_call = ast.is<Nodecl::OmpSs::TaskCall>();
if (is_task_call)
{
Nodecl::List nested_env;
Nodecl::FunctionCall func_call = ast.as<Nodecl::OmpSs::TaskCall>().get_call().as<Nodecl::FunctionCall>();
bool verified = analyze_nested_call(func_call, nested_env);
if (!verified)
{
std::cerr << "lint-debug: task in " << ast.get_locus_str()
<< " cannot be verified." << std::endl;
return false;
}
else
{
Analysis::NodeclSet user_in_set, user_out_set, user_inout_set;
get_dependency_sets(env, user_in_set, user_out_set, user_inout_set);
Analysis::NodeclSet analyzed_in_set, analyzed_out_set, analyzed_inout_set;
get_dependency_sets(nested_env, analyzed_in_set, analyzed_out_set, analyzed_inout_set);
Nodecl::NodeclBase fp_clause = env.find_first<Nodecl::OpenMP::Firstprivate>();
purge_firstprivate(fp_clause, analyzed_in_set, analyzed_out_set, analyzed_inout_set);
verified = !check_lint_task_dependencies(
user_in_set, analyzed_in_set,
user_out_set, analyzed_out_set,
user_inout_set, analyzed_inout_set,
ast.get_locus_str());
if (verified)
{
std::cerr << "lint-debug: adding verified clause to task in "
<< ast.get_locus_str() << std::endl;
Nodecl::OmpSs::LintVerified verified_clause =
Nodecl::OmpSs::LintVerified::make(
Nodecl::IntegerLiteral::make(
TL::Type::get_size_t_type(),
const_value_get_signed_int(1)
)
);
ast.as<Nodecl::OmpSs::TaskCall>().get_environment().as<Nodecl::List>().append(verified_clause);
n->add_verified_clause(verified_clause);
append_to_path(path, ast);
replace_parameters_by_arguments(func_call.get_called().get_symbol().get_function_parameters(),
func_call.get_arguments().as<Nodecl::List>(),
env);
append_to_attrs(attrs, env);
return true;
}
else
{
std::cerr << "lint-debug: task in " << ast.get_locus_str()
<< " cannot be verified." << std::endl;
return false;
}
}
}
LintAttributes* graph_attrs = new LintAttributes();
ObjectList<Nodecl::NodeclBase> graph_path;
bool nested_continue_outer_levels = true;
Analysis::Utils::InductionVarList new_ind_vars;
bool verified = analyze_path(n->get_graph_exit_node(),
graph_path, graph_attrs, new_ind_vars,
nested_continue_outer_levels);
verified = verified && nested_continue_outer_levels;
continue_outer_levels = continue_outer_levels && nested_continue_outer_levels;
for (Analysis::Utils::InductionVarList::iterator it = new_ind_vars.begin(); it != new_ind_vars.end(); ++it)
{
Analysis::Utils::InductionVarList::iterator ita = ind_vars.begin();
for ( ; ita != ind_vars.end(); ++ita)
{
if (Nodecl::Utils::structurally_equal_nodecls((*ita)->get_variable(), (*it)->get_variable(),
true))
break;
}
if (ita == ind_vars.end())
ind_vars.append(*it);
}
if (!verified)
{
std::cerr << "lint-warning: task in " << ast.get_locus_str()
<< " cannot be verified." << std::endl;
wrap_statements(graph_path, graph_attrs);
return false;
}
if (Nodecl::Utils::nodecl_contains_nodecl_of_kind<Nodecl::ReturnStatement>(ast))
{
std::cerr << "lint-warning: task in " << ast.get_locus_str()
<< " can be verified but it contains a return statement,"
<< " so no verified clause is added." << std::endl;
return false;
}
Analysis::NodeclSet dep_in_set, dep_out_set, dep_inout_set;
get_dependency_sets(env, dep_in_set, dep_out_set, dep_inout_set);
Nodecl::NodeclBase dep_weak_in_clause = env.find_first<Nodecl::OmpSs::DepWeakIn>();
Analysis::NodeclSet dep_weak_in_set;
if (!dep_weak_in_clause.is_null())
{
Nodecl::List dep_weak_in_list = dep_weak_in_clause.as<Nodecl::OmpSs::DepWeakIn>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_weak_in_list.begin(); it != dep_weak_in_list.end(); ++it)
dep_weak_in_set.insert(*it);
}
Nodecl::NodeclBase dep_weak_out_clause = env.find_first<Nodecl::OmpSs::DepWeakOut>();
Analysis::NodeclSet dep_weak_out_set;
if (!dep_weak_out_clause.is_null())
{
Nodecl::List dep_weak_out_list = dep_weak_out_clause.as<Nodecl::OmpSs::DepWeakOut>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_weak_out_list.begin(); it != dep_weak_out_list.end(); ++it)
dep_weak_out_set.insert(*it);
}
Nodecl::NodeclBase dep_weak_inout_clause = env.find_first<Nodecl::OmpSs::DepWeakInout>();
Analysis::NodeclSet dep_weak_inout_set;
if (!dep_weak_inout_clause.is_null())
{
Nodecl::List dep_weak_inout_list = dep_weak_inout_clause.as<Nodecl::OmpSs::DepWeakInout>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = dep_weak_inout_list.begin(); it != dep_weak_inout_list.end(); ++it)
dep_weak_inout_set.insert(*it);
}
Nodecl::NodeclBase lint_alloc_clause = env.find_first<Nodecl::OmpSs::LintAlloc>();
Analysis::NodeclSet lint_alloca_set;
if (!lint_alloc_clause.is_null())
{
Nodecl::List lint_alloc_list = lint_alloc_clause.as<Nodecl::OmpSs::LintAlloc>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = lint_alloc_list.begin(); it != lint_alloc_list.end(); ++it)
lint_alloca_set.insert(*it);
}
Nodecl::NodeclBase lint_free_clause = env.find_first<Nodecl::OmpSs::LintFree>();
Analysis::NodeclSet lint_free_set;
if (!lint_free_clause.is_null())
{
Nodecl::List lint_free_list = lint_free_clause.as<Nodecl::OmpSs::LintFree>().get_exprs().as<Nodecl::List>();
for (Nodecl::List::iterator it = lint_free_list.begin(); it != lint_free_list.end(); ++it)
lint_free_set.insert(*it);
}
Analysis::NodeclSet& attrs_in = graph_attrs->get_in();
Analysis::NodeclSet& attrs_out = graph_attrs->get_out();
Analysis::NodeclSet& attrs_inout = graph_attrs->get_inout();
Nodecl::NodeclBase fp_clause = env.find_first<Nodecl::OpenMP::Firstprivate>();
purge_firstprivate(fp_clause, attrs_in, attrs_out, attrs_inout);
for (Analysis::NodeclSet::iterator it = attrs_inout.begin(); it != attrs_inout.end(); ++it)
{
Analysis::NodeclSet::iterator iti = attrs_in.find(*it);
if (iti != attrs_in.end())
{
attrs_in.erase(iti);
}
Analysis::NodeclSet::iterator ito = attrs_out.find(*it);
if (ito != attrs_out.end())
{
attrs_out.erase(ito);
}
}
for (Analysis::NodeclSet::iterator iti = dep_in_set.begin(); iti != dep_in_set.end(); )
{
Analysis::NodeclSet::iterator ito = attrs_out.find(*iti);
if (ito != attrs_out.end())
{
attrs_inout.insert(*ito);
attrs_out.erase(ito);
iti = dep_in_set.erase(iti);
}
else
{
++iti;
}
}
if (!dep_weak_in_set.empty() || !dep_weak_out_set.empty() || !dep_weak_inout_set.empty())
{
std::cerr << "lint-debug: task at " << ast.get_locus_str()
<< " cannot be verified." << std::endl;
return false;
}
std::string task_locus = n->get_graph_related_ast().get_locus_str();
bool wrong_dependencies = check_lint_task_dependencies(dep_in_set, attrs_in,
dep_out_set, attrs_out,
dep_inout_set, attrs_inout,
task_locus);
Analysis::NodeclSet attrs_alloca = graph_attrs->get_alloc();
Analysis::NodeclSet overprotective_alloca, missing_alloca;
std::multimap<Nodecl::NodeclBase, Nodecl::NodeclBase,
Nodecl::Utils::Nodecl_structural_less> mistaken_alloca;
check_lint_task_memory(lint_alloca_set, attrs_alloca, overprotective_alloca, missing_alloca, mistaken_alloca);
report_overprotective(overprotective_alloca, "alloca", task_locus);
report_missing(missing_alloca, "alloca", task_locus);
report_mistaken(mistaken_alloca, "free", task_locus);
Analysis::NodeclSet attrs_free = graph_attrs->get_free();
Analysis::NodeclSet overprotective_free, missing_free;
std::multimap<Nodecl::NodeclBase, Nodecl::NodeclBase,
Nodecl::Utils::Nodecl_structural_less> mistaken_free;
check_lint_task_memory(lint_free_set, attrs_free, overprotective_free, missing_free, mistaken_free);
report_overprotective(overprotective_free, "free", task_locus);
report_missing(missing_free, "free", task_locus);
report_mistaken(mistaken_free, "free", task_locus);
if (wrong_dependencies
|| !missing_alloca.empty() || !missing_free.empty()
|| !mistaken_alloca.empty() || !mistaken_free.empty())
{
return false;
}
std::cerr << "lint-debug: adding verified clause to task in "
<< n->get_graph_related_ast().get_locus_str() << std::endl;
Nodecl::OpenMP::Task task_ast = n->get_graph_related_ast().as<Nodecl::OpenMP::Task>();
Nodecl::OmpSs::LintVerified verified_clause =
Nodecl::OmpSs::LintVerified::make(
Nodecl::IntegerLiteral::make(
TL::Type::get_size_t_type(),
const_value_get_signed_int(1)
)
);
task_ast.get_environment().as<Nodecl::List>().append(verified_clause);
n->add_verified_clause(verified_clause);
append_to_path(path, ast);
attrs->add_attrs(graph_attrs);
return true;
}
static void replace_graph_attributes(
Analysis::NodeclSet& attr_set,
const Analysis::Utils::InductionVarList& ind_vars)
{
Analysis::NodeclSet new_attr_set;
for (Analysis::NodeclSet::iterator it = attr_set.begin(); it != attr_set.end(); )
{
if (it->is<Nodecl::ArraySubscript>())
{
Nodecl::NodeclBase v = Nodecl::Utils::deep_copy(*it, it->retrieve_context());
bool replaced = replace_ivs_with_ranges(v, ind_vars);
if (replaced)
{
it = attr_set.erase(it);
new_attr_set.insert(v);
}
else
{
++it;
}
}
else
{
++it;
}
}
if (!new_attr_set.empty())
attr_set.insert(new_attr_set.begin(), new_attr_set.end());
}
static bool all_symbols_alive(LintAttributes* attrs, Scope sc)
{
Analysis::NodeclSet& in = attrs->get_in();
Analysis::NodeclSet& out = attrs->get_out();
Analysis::NodeclSet& inout = attrs->get_inout();
Analysis::NodeclSet& lint_alloc = attrs->get_alloc();
Analysis::NodeclSet& lint_free = attrs->get_free();
return all_symbols_alive_rec(in, sc) && all_symbols_alive_rec(out, sc)
&& all_symbols_alive_rec(inout, sc) && all_symbols_alive_rec(lint_alloc, sc)
&& all_symbols_alive_rec(lint_free, sc);
}
static bool propagate_induction_variables(
const Analysis::Utils::InductionVarList& new_ind_vars,
Analysis::Utils::InductionVarList& ind_vars)
{
bool continue_outer_levels = true;
for (Analysis::Utils::InductionVarList::const_iterator it = new_ind_vars.begin(); it != new_ind_vars.end(); ++it)
{
Analysis::Utils::InductionVarList::iterator ita = ind_vars.begin();
for ( ; ita != ind_vars.end(); ++ita)
{
if (Nodecl::Utils::structurally_equal_nodecls((*ita)->get_variable(), (*it)->get_variable(),
true))
break;
}
if (ita != ind_vars.end())
{
if (!Analysis::Utils::nodecl_set_difference((*ita)->get_lb(), (*it)->get_lb()).empty()
|| !Analysis::Utils::nodecl_set_difference((*ita)->get_ub(), (*it)->get_ub()).empty()
|| !Nodecl::Utils::structurally_equal_nodecls((*ita)->get_increment(), (*it)->get_increment(), true))
{
continue_outer_levels = false;
}
}
else
{
ind_vars.append(*it);
}
}
return continue_outer_levels;
}
bool OssLint::analyze_graph_node(
Analysis::Node* n,
ObjectList<Nodecl::NodeclBase>& path,
LintAttributes* attrs,
Analysis::Utils::InductionVarList& ind_vars,
bool& continue_outer_levels)
{
Nodecl::NodeclBase ast = n->get_graph_related_ast();
if (n->is_loop_node())
{
Analysis::Utils::InductionVarList tmp = n->get_induction_variables();
for (Analysis::Utils::InductionVarList::iterator it = tmp.begin(); it != tmp.end(); ++it)
{
if ((*it)->get_lb().size() != 1
|| (*it)->get_ub().size() != 1)
{
std::cerr << "lint-debug: undefined IV boundaries loop at " << ast.get_locus_str()
<< ". This prevents Lint optimization." << std::endl;
return false;
}
}
}
else if (n->is_while_loop())
{
if (n->get_condition_node()->is_graph_node())
{
std::cerr << "lint-debug: complex while loop at " << ast.get_locus_str()
<< ". This prevents Lint optimization." << std::endl;
return false;
}
else
{
Analysis::NodeclList cond_stmt = n->get_condition_node()->get_statements();
if (cond_stmt.size() > 1
|| !Nodecl::Utils::nodecl_is_comparison_op(cond_stmt.at(0)))
{
std::cerr << "lint-debug: complex while loop at " << ast.get_locus_str()
<< ". This prevents Lint optimization." << std::endl;
return false;
}
}
}
Analysis::Utils::InductionVarList new_ind_vars;
bool verified = false;
if (n->is_ifelse_statement())
{   
Analysis::Node* exit = n->get_graph_exit_node();
exit->set_visited(true);
ObjectList<Analysis::Node*> exit_parents = exit->get_parents();
if (exit_parents.size() > 2)
{
std::cerr << "lint-debug: unexpected if-statement shape."
<< " More than two parents for exit node in statement "
<< ast.get_locus_str()
<< ". Not verifying!" << std::endl;
continue_outer_levels = false;
return false;
}
Analysis::Node* entry = n->get_graph_entry_node();
ObjectList<Analysis::Node*> entry_children = entry->get_children();
if (entry_children.size() != 1)
{
std::cerr << "lint-debug: unexpected if-statement shape."
<< " More than one children for entry node in statement "
<< ast.get_locus_str()
<< ". Not verifying!" << std::endl;
continue_outer_levels = false;
return false;
}
entry->set_visited(true);
Analysis::Node* cond = entry_children.at(0);
ObjectList<Nodecl::NodeclBase> path_cond;
LintAttributes* attrs_cond = new LintAttributes();  
bool nested_continue_outer_levels_cond = true;
bool verified_cond = analyze_path(cond,
path_cond, attrs_cond, new_ind_vars,
nested_continue_outer_levels_cond);
if (exit_parents.size() != 2)
{
std::cerr << "lint-debug: if-statement node at " << ast.get_locus_str()
<< " contains return statements."
<< " This prevents Lint optimization." << std::endl;
continue_outer_levels = false;
return false;
}
else
{
Analysis::Node* branch_a = exit_parents.at(0);
ObjectList<Nodecl::NodeclBase> path_a;
LintAttributes* attrs_a = new LintAttributes();
bool nested_continue_outer_levels_a = true;
bool verified_a = analyze_path(branch_a,
path_a, attrs_a, new_ind_vars,
nested_continue_outer_levels_a);
Analysis::Node* branch_b = exit_parents.at(1);
if (branch_b->is_visited())
{   
if (verified_a && nested_continue_outer_levels_a)
{
if (!verified_cond)
{
wrap_statements(path_a, attrs_a);
continue_outer_levels = false;
verified = false;
}
else
{
verified = true;
attrs->add_attrs(attrs_a);
attrs->add_attrs(attrs_cond);
path.append(ast);
}
}
else
{
if (verified_a)
{
wrap_statements(path_a, attrs_a);
}
continue_outer_levels = false;
verified = false;
}
}
else
{
ObjectList<Nodecl::NodeclBase> path_b;
LintAttributes* attrs_b = new LintAttributes();
bool nested_continue_outer_levels_b = true;
bool verified_b = analyze_path(branch_b,
path_b, attrs_b, new_ind_vars,
nested_continue_outer_levels_b);
if (verified_a && verified_b
&& nested_continue_outer_levels_a
&& nested_continue_outer_levels_b)
{
{   
Nodecl::NodeclBase encl_list = Nodecl::Utils::get_enclosing_list(ast);
if (!encl_list.is_null())
{
Scope sc = encl_list.retrieve_context();
bool _all_symbols_alive = all_symbols_alive(attrs_a, sc) && all_symbols_alive(attrs_b, sc);
bool defines_previous_variables = new_attributes_define_previous_attributes( attrs,  attrs_a);
if (!_all_symbols_alive)
{
if (!encl_list.get_parent().is<Nodecl::CompoundStatement>()
|| !encl_list.get_parent().get_parent().is<Nodecl::List>()
|| !encl_list.get_parent().get_parent().get_parent().is<Nodecl::Context>()
|| !encl_list.get_parent().get_parent().get_parent().get_parent().is<Nodecl::FunctionCode>())
std::cerr << "lint-warning: variables declared at " << ast.get_locus_str()
<< " do not allow the insertion of a lint construct. "
<< "Consider promoting the declaration to upper levels "
<< "for a better Lint optimization." << std::endl;
wrap_statements(path_a, attrs_a);
wrap_statements(path_b, attrs_b);
continue_outer_levels = false;
verified = false;
}
else
{
if (defines_previous_variables)
{
wrap_statements(path, attrs);
continue_outer_levels = false;
}
attrs->add_attrs(attrs_a);
attrs->add_attrs(attrs_b);
attrs->add_attrs(attrs_cond);
append_to_path(path, ast);
verified = true;
}
}
}
}
else
{
if (verified_a)
{
wrap_statements(path_a, attrs_a);
}
if (verified_b)
{
wrap_statements(path_b, attrs_b);
}
continue_outer_levels = false;
verified = false;
}
}
continue_outer_levels = continue_outer_levels && propagate_induction_variables(new_ind_vars, ind_vars);
return verified;
}
}
LintAttributes* graph_attrs = new LintAttributes();
ObjectList<Nodecl::NodeclBase> graph_path;
bool nested_continue_outer_levels = true;
verified = analyze_path(n->get_graph_exit_node(),
graph_path, graph_attrs, new_ind_vars,
nested_continue_outer_levels);
verified = verified && nested_continue_outer_levels;
verified = verified && !Nodecl::Utils::nodecl_contains_nodecl_of_kind<Nodecl::ReturnStatement>(ast);
continue_outer_levels = continue_outer_levels && nested_continue_outer_levels;
const ObjectList<Analysis::Node*> parents = n->get_parents();
bool loop_iv_outside = n->is_loop_node()
&& parents.size() == 1 && parents[0]->has_statements()
&& parents[0]->get_statements().size() == 1
&& Nodecl::Utils::nodecl_contains_nodecl_by_pointer( ast,
parents[0]->get_statements().at(0));
if (n->is_loop_node())
{
new_ind_vars.append(n->get_induction_variables());
}
continue_outer_levels = continue_outer_levels && propagate_induction_variables(new_ind_vars, ind_vars);
if (verified)
{
if (nested_continue_outer_levels)
{
Nodecl::NodeclBase encl_list = Nodecl::Utils::get_enclosing_list(ast);
if (!encl_list.is_null())
{
Scope sc = encl_list.retrieve_context();
bool _all_symbols_alive = !n->is_context_node()
|| all_symbols_alive(graph_attrs, sc);
bool loop_iv_initializer = loop_iv_outside
&& parents[0]->get_statements()[0].is<Nodecl::ObjectInit>();
bool defines_previous_variables = new_attributes_define_previous_attributes( attrs,  graph_attrs);
if (!_all_symbols_alive || loop_iv_initializer)
{
verified = false;
continue_outer_levels = false;
if (!encl_list.get_parent().is<Nodecl::CompoundStatement>()
|| !encl_list.get_parent().get_parent().is<Nodecl::List>()
|| !encl_list.get_parent().get_parent().get_parent().is<Nodecl::Context>()
|| !encl_list.get_parent().get_parent().get_parent().get_parent().is<Nodecl::FunctionCode>())
std::cerr << "lint-warning: variables declared at " << ast.get_locus_str()
<< " do not allow the insertion of a lint construct. "
<< "Consider promoting the declaration to upper levels "
<< "for a better Lint optimization." << std::endl;
wrap_statements(graph_path, graph_attrs);
}
else
{
if (defines_previous_variables)
{
wrap_statements(path, attrs);
continue_outer_levels = false;
}
else if (n->is_loop_node()
|| (n->is_context_node() && n->get_outer_node()->is_loop_node()))
{   
Analysis::NodeclSet& graph_in = graph_attrs->get_in();
Analysis::NodeclSet& graph_out = graph_attrs->get_out();
Analysis::NodeclSet& graph_inout = graph_attrs->get_inout();
replace_graph_attributes(graph_in, ind_vars);
replace_graph_attributes(graph_out, ind_vars);
replace_graph_attributes(graph_inout, ind_vars);
if (loop_iv_outside)
{
bool init_verified = analyze_stmt(parents[0], graph_path, graph_attrs, ind_vars, continue_outer_levels);
if (!init_verified)
{
wrap_statements(graph_path, graph_attrs);
verified = false;
continue_outer_levels = false;
LintAttributes* new_attrs = new LintAttributes();
ObjectList<Nodecl::NodeclBase> new_path;
ObjectList<Analysis::Node*> grand_parents = parents[0]->get_parents();
for (ObjectList<Analysis::Node*>::iterator it = grand_parents.begin();
it != grand_parents.end(); ++it)
{
analyze_path(*it, new_path, new_attrs, ind_vars, continue_outer_levels);
}
return verified;
}
}
}
attrs->add_attrs(graph_attrs);
if (ast.is<Nodecl::FunctionCode>())
{
path.append(graph_path);
}
else
{
append_to_path(path, ast);
}
}
}
else {} 
}
else
{
wrap_statements(graph_path, graph_attrs);
}
}
propagate_induction_variables(new_ind_vars, ind_vars);
return verified;
}
bool OssLint::analyze_simple_graph_node(Analysis::Node* n,
ObjectList<Nodecl::NodeclBase>& path,
LintAttributes* attrs,
Analysis::Utils::InductionVarList& ind_vars,
bool& continue_outer_levels)
{
bool is_loop_condition =
n->get_outer_node()->is_for_loop()
&& n->get_children().size() == 2
&& (n->get_exit_edges().at(0)->is_true_edge()
|| n->get_exit_edges().at(0)->is_false_edge());
bool is_loop_increment =
n->get_outer_node()->is_for_loop()
&& (n->get_children().size() == 1)
&& (n->get_children().at(0)->get_children().size() == 2)
&& (n->get_children().at(0)->get_exit_edges().at(0)->is_true_edge()
|| n->get_children().at(0)->get_exit_edges().at(0)->is_false_edge());
bool is_loop_init = (n->get_children().size() == 1)
&& n->get_children().at(0)->is_for_loop()
&& n->get_children().at(0)->get_graph_related_ast().is<Nodecl::ForStatement>()
&& Nodecl::Utils::nodecl_contains_nodecl_by_pointer(
n->get_children().at(0)->get_graph_related_ast().as<Nodecl::ForStatement>().get_loop_header(),
n->get_graph_related_ast());
bool path_contains_task = false;
for (ObjectList<Nodecl::NodeclBase>::iterator it = path.begin();
it != path.end() && !path_contains_task; ++it)
{
if (it->is<Nodecl::OpenMP::Task>()
|| it->is<Nodecl::OmpSs::TaskCall>())
{
path_contains_task = true;
}
}
LintAttributes* new_attrs = new LintAttributes();
bool verified = update_attributes(n, new_attrs, ind_vars);
bool propagate = true;
if ((!is_loop_condition && !is_loop_increment)
|| path_contains_task)
{
if (!verified || new_attributes_define_previous_attributes( attrs,  new_attrs))
{
if (!is_loop_condition && !is_loop_increment)
propagate = false;
continue_outer_levels = false;
}
}
if (!propagate)
{
wrap_statements(path, attrs);
}
if (verified)
{
append_to_path(path, n->get_graph_related_ast());
if (!is_loop_init || path.size()>0)
{   
attrs->add_attrs(new_attrs);
}
}
return verified;
}
bool OssLint::analyze_stmt(Analysis::Node* n,
ObjectList<Nodecl::NodeclBase>& path,
LintAttributes* attrs,
Analysis::Utils::InductionVarList& ind_vars,
bool& continue_outer_levels)
{
n->set_visited(true);
bool verified = false;
if (n->is_omp_task_creation_node())
{
const ObjectList<Analysis::Node*>& children = n->get_children();
for (ObjectList<Analysis::Node*>::const_iterator it = children.begin();
it != children.end(); ++it)
{
if (!(*it)->is_visited())
{
verified = analyze_stmt(*it, path, attrs, ind_vars,
continue_outer_levels);
if (!verified)
continue_outer_levels = false;
}
}
}
else if (n->is_graph_node())
{
if (n->is_omp_task_node())
{
verified = analyze_task_node(n, path, attrs, ind_vars,
continue_outer_levels);
}
else if (n->is_ompss_lint_node())
{
append_to_path(path, n->get_graph_related_ast());
append_to_attrs(attrs, n->get_graph_related_ast().as<Nodecl::OmpSs::Lint>().get_environment().as<Nodecl::List>());
verified = true;
}
else if (n->is_split_statement())
{
verified = analyze_simple_graph_node(n, path, attrs, ind_vars,
continue_outer_levels);
}
else
{
verified = analyze_graph_node(n, path, attrs, ind_vars,
continue_outer_levels);
if (!verified)
continue_outer_levels = false;
}
}
else
{
if (n->get_statements().empty())
{
verified = true;
}
else
{
verified = analyze_simple_node(n, path, attrs, ind_vars,
continue_outer_levels);
}
}
return verified;
}
static bool in_loop(Analysis::Node* n)
{
Analysis::Node* outer = n->get_outer_node();
while (outer != NULL)
{
if (outer->is_loop_node())
return true;
if (outer->is_ifelse_statement()
|| outer->is_switch_statement()
|| outer->is_switch_case_node())
return false;
outer = outer->get_outer_node();
}
return false;
}
bool OssLint::analyze_path(Analysis::Node* n,
ObjectList<Nodecl::NodeclBase>& path,
LintAttributes* attrs,
Analysis::Utils::InductionVarList& ind_vars,
bool& continue_outer_levels)
{
if (n->is_visited())
return true;
if (n->is_omp_task_node())
return true;
bool verified = true;
LintAttributes* all_attrs = attrs;
const ObjectList<Analysis::Edge*>& exits = n->get_exit_edges();
if (!n->is_omp_task_creation_node()
&& exits.size() > 1)
{
if (exits.size() == 2
&& ((exits[0]->is_false_edge() && exits[1]->is_true_edge())
|| (exits[0]->is_true_edge() && exits[1]->is_false_edge()))
&& in_loop(n))
{
const ObjectList<Analysis::Edge*>& entries = n->get_entry_edges();
Analysis::Node* back_parent =
entries[0]->is_back_edge() ? entries[0]->get_source()
: entries[1]->get_source();
Analysis::Node* non_back_parent =
entries[0]->is_back_edge() ? entries[1]->get_source()
: entries[0]->get_source();
ERROR_CONDITION(non_back_parent->is_visited(),
"Loop condition node %d has non back parent already visited.",
n->get_id());
verified = analyze_stmt(n, path, attrs, ind_vars, continue_outer_levels);
if (verified)
{
verified = analyze_path(back_parent, path, attrs, ind_vars,
continue_outer_levels);
if (verified)
{
verified = analyze_path(non_back_parent, path, attrs, ind_vars,
continue_outer_levels);
}
else
{
ObjectList<Nodecl::NodeclBase> new_path;
LintAttributes* new_attrs = new LintAttributes();
analyze_path(non_back_parent, new_path, new_attrs, ind_vars,
continue_outer_levels);
}
}
else
{
ObjectList<Nodecl::NodeclBase> new_path;
LintAttributes* new_attrs = new LintAttributes();
analyze_path(back_parent, new_path, new_attrs, ind_vars,
continue_outer_levels);
}
return verified;
}
else if (!n->get_outer_node()->is_ifelse_statement()
|| (n->get_parents().size()==1 && !n->get_parents().at(0)->is_entry_node()))
{
bool all_children_visited = true;
for (ObjectList<Analysis::Edge*>::const_iterator it = exits.begin();
it != exits.end(); ++it)
{
if (!(*it)->get_target()->is_visited())
{
all_children_visited = false;
break;
}
}
if (all_children_visited)
{
if (_pending_attrs.find(n) != _pending_attrs.end())
{
all_attrs->add_attrs(_pending_attrs[n]);
_pending_attrs.erase(n);
}
}
else
{
if (_pending_attrs.find(n) != _pending_attrs.end())
_pending_attrs[n]->add_attrs(attrs);
else
_pending_attrs[n] = attrs;
return false;
}
}
}
verified = analyze_stmt(n, path, all_attrs, ind_vars, continue_outer_levels);
if (n->is_entry_node())
{   
if (!continue_outer_levels)
{
if (n->get_outer_node()->is_ifelse_statement())
{
attrs->clear();
path.clear();
}
else
{
wrap_statements(path, all_attrs);
}
}
else if ((n->get_outer_node()->is_context_node()
&& n->get_outer_node()->get_outer_node()->is_function_code_node())
|| n->get_outer_node()->is_extended_graph_node())
{
wrap_statements(path, all_attrs);
}
}
else if (n->is_context_node() && n->get_outer_node()->is_loop_node() && !continue_outer_levels)
{   
attrs->clear();
path.clear();
}
else
{
if (n->is_return_node())
{
Analysis::Node* outer = n->get_outer_node();
while (outer != NULL)
{
if (outer->is_ompss_lint_node()
|| (outer->is_omp_task_node() && outer->has_verified_clause()))
{
return true;
}
outer = outer->get_outer_node();
}
}
ObjectList<Analysis::Node*> parents = n->get_parents();
if (verified)
{
for (ObjectList<Analysis::Node*>::iterator it = parents.begin();
it != parents.end(); ++it)
{
if ((*it)->is_omp_task_node())
continue;
analyze_path(*it, path, all_attrs, ind_vars, continue_outer_levels);
}
}
else
{
continue_outer_levels = false;
wrap_statements(path, all_attrs);
for (ObjectList<Analysis::Node*>::iterator it = parents.begin();
it != parents.end(); ++it)
{
if ((*it)->is_omp_task_node())
continue;
LintAttributes* new_attrs = new LintAttributes();
ObjectList<Nodecl::NodeclBase> new_path;
analyze_path(*it, new_path, new_attrs, ind_vars, continue_outer_levels);
}
}
}
return verified;
}
void OssLint::oss_lint_analyze_all(TL::Analysis::ExtensibleGraph* pcfg)
{
if (analyzed_pcfgs.find(pcfg) != analyzed_pcfgs.end())
return;
pcfg->print_graph_to_dot();
Analysis::Node* graph = pcfg->get_graph();
Analysis::Node* exit = graph->get_graph_exit_node();
ObjectList<Analysis::Node*> parents = exit->get_parents();
for (ObjectList<Analysis::Node*>::iterator it = parents.begin(); it != parents.end(); ++it)
{
if ((*it)->is_omp_task_node())
continue;
LintAttributes* attrs = new LintAttributes();
ObjectList<Nodecl::NodeclBase> path;
Analysis::Utils::InductionVarList ind_vars;
bool continue_outer_levels = true;
analyze_path(*it, path, attrs, ind_vars, continue_outer_levels);
}
analyzed_pcfgs.insert(pcfg);
Analysis::ExtensibleGraph::clear_visits(graph);
}
static void gather_induction_vars(
Analysis::Node* n,
Analysis::Utils::InductionVarList& ind_vars)
{
Analysis::Node* outer =
Analysis::ExtensibleGraph::get_task_creation_from_task(n)->get_outer_node();
while (outer != NULL)
{
if (outer->is_loop_node())
{
Analysis::Utils::InductionVarList outer_ind_vars = outer->get_induction_variables();
for (Analysis::Utils::InductionVarList::iterator it = outer_ind_vars.begin();
it != outer_ind_vars.end(); ++it)
{
Analysis::Utils::InductionVarList::iterator ita = ind_vars.begin();
for ( ; ita != ind_vars.end(); ++ita)
{
if (Nodecl::Utils::structurally_equal_nodecls((*ita)->get_variable(), (*it)->get_variable(),
true))
break;
}
if (ita == ind_vars.end())
ind_vars.append(*it);
}
}
outer = outer->get_outer_node();
}
}
void OssLint::oss_lint_analyze_tasks(TL::Analysis::ExtensibleGraph* pcfg)
{
ObjectList<Analysis::Node*> tasks = pcfg->get_tasks_list();
for (ObjectList<Analysis::Node*>::iterator it = tasks.begin(); it != tasks.end(); ++it)
{
if (analyzed_tasks.find(*it) != analyzed_tasks.end())
continue;
ObjectList<Nodecl::NodeclBase> path;
LintAttributes* attrs = new LintAttributes();
Analysis::Utils::InductionVarList ind_vars;
gather_induction_vars(*it, ind_vars);
bool continue_outer_levels;
analyze_task_node(*it, path, attrs, ind_vars, continue_outer_levels);
analyzed_tasks.insert(*it);
}
}
}
}
EXPORT_PHASE(TL::Nanos6::OssLint)
