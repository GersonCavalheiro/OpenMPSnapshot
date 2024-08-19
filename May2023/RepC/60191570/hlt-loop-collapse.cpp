#include "hlt-loop-collapse.hpp"
#include "hlt-utils.hpp"
#include "tl-nodecl-utils.hpp"
#include "cxx-cexpr.h"
#include "cxx-diagnostic.h"
namespace TL { namespace HLT {
LoopCollapse::LoopCollapse()
: _loop(), _transformation(), _collapse_factor(-1)
{
}
LoopCollapse& LoopCollapse::set_loop(Nodecl::NodeclBase loop)
{
_loop = loop;
return *this;
}
LoopCollapse& LoopCollapse::set_collapse_factor(int collapse_factor)
{
_collapse_factor = collapse_factor;
return *this;
}
LoopCollapse& LoopCollapse::set_pragma_context(const TL::Scope& context)
{
_pragma_context = context;
return *this;
}
Nodecl::NodeclBase LoopCollapse::get_post_transformation_stmts() const
{
return _post_transformation_stmts;
}
TL::ObjectList<TL::Symbol> LoopCollapse::get_omp_capture_symbols() const
{
return _omp_capture_symbols;
}
namespace {
struct LoopInfo
{
TL::Symbol induction_var;
Nodecl::NodeclBase lower_bound;
Nodecl::NodeclBase upper_bound;
Nodecl::NodeclBase step;
};
void check_loop(Nodecl::NodeclBase node, int collapse_factor)
{
TL::ObjectList<TL::Symbol> induction_variables;
for (int i = 0; i < collapse_factor; ++i)
{
if (!node.is<Nodecl::ForStatement>())
fatal_printf_at(
node.get_locus(),
"Trying to collapse %d 'For' loop(s) but only %d loop(s) found\n",
collapse_factor, i + 1);
TL::ForStatement for_stmt(node.as<Nodecl::ForStatement>());
if (!for_stmt.is_omp_valid_loop())
fatal_printf_at(
node.get_locus(),
"Trying to collapse an invalid 'For' loop in nesting level (%d)\n",
i + 1);
struct LoopControlSymbolVisitor : public Nodecl::ExhaustiveVisitor<void>
{
const TL::ObjectList<TL::Symbol>& _symbol_list;
bool _containsAnySymbol;
LoopControlSymbolVisitor(const TL::ObjectList<TL::Symbol>& symbol_list)
: _symbol_list(symbol_list), _containsAnySymbol(false)
{ }
void visit(const Nodecl::Symbol& node)
{
TL::Symbol sym = node.get_symbol();
if (!sym.is_variable())
return;
walk(sym.get_value());
for (TL::ObjectList<TL::Symbol>::const_iterator it = _symbol_list.begin();
it != _symbol_list.end();
it++)
{
if (sym == *it)
{
_containsAnySymbol = true;
break;
}
}
}
};
LoopControlSymbolVisitor loopControlVisitor(induction_variables);
loopControlVisitor.walk(for_stmt.get_loop_header());
if (loopControlVisitor._containsAnySymbol)
{
fatal_printf_at(
node.get_locus(),
"Trying to collapse a non-canonical loop: induction variable found in nested loop header\n");
}
induction_variables.append(for_stmt.get_induction_variable());
node = for_stmt.get_statement()
.as<Nodecl::List>().front()
.as<Nodecl::Context>().get_in_context();
ERROR_CONDITION(node.is_null(),
"Empty context within 'For'", 0);
node = node
.as<Nodecl::List>().front();
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
node = node
.as<Nodecl::CompoundStatement>().get_statements();
if (i < collapse_factor - 1)
{
if (node.is_null())
fatal_printf_at(
node.get_locus(),
"Trying to collapse %d 'For' loop(s) but only %d loop(s) found\n",
collapse_factor, i + 1);
node = node.as<Nodecl::List>().front();
if (!node.is<Nodecl::Context>())
fatal_printf_at(
node.get_locus(),
"Trying to collapse %d 'For' loop(s) but only %d loop(s) found\n",
collapse_factor, i + 1);
node = node.as<Nodecl::Context>().get_in_context();
ERROR_CONDITION(node.is_null(),
"Empty context within 'Compound statement'", 0);
node = node.as<Nodecl::List>().front();
}
}
}
}
void compute_collapse_statements(
int collapse_factor,
TL::Symbol collapse_induction_var,
Nodecl::Context context_innermost_loop,
TL::ObjectList<LoopInfo>& loop_info,
TL::Scope& collapse_scope,
TL::Scope& loop_statements_scope,
Nodecl::List& collapse_statements,
Nodecl::List& new_loop_statements,
Nodecl::NodeclBase& condition_bound,
TL::ObjectList<TL::Symbol>& omp_capture_symbols)
{
if (IS_CXX_LANGUAGE)
{
collapse_statements.append(
Nodecl::CxxDef::make(
Nodecl::NodeclBase::null(),
collapse_induction_var));
}
Nodecl::Utils::SimpleSymbolMap induction_var_map;
Nodecl::NodeclBase num_elem_in_nested_loops =
const_value_to_nodecl_with_basic_type(
const_value_get_unsigned_long_long_int(1),
TL::Type::get_unsigned_long_long_int_type().get_internal_type());
for (int i = collapse_factor - 1; i >= 0; --i)
{
std::stringstream step_ss;
step_ss << "collapse_" << i << "_step";
TL::Symbol step_var = collapse_scope.new_symbol(step_ss.str());
symbol_entity_specs_set_is_user_declared(step_var.get_internal_symbol(), 1);
step_var.get_internal_symbol()->kind = SK_VARIABLE;
step_var.set_type(loop_info[i].step.get_type().no_ref().get_unqualified_type());
step_var.set_value(loop_info[i].step);
omp_capture_symbols.insert(step_var);
std::stringstream num_elem_ss;
num_elem_ss << "collapse_" << i << "_num_elements";
TL::Symbol num_elem_var = collapse_scope.new_symbol(num_elem_ss.str());
symbol_entity_specs_set_is_user_declared(num_elem_var.get_internal_symbol(), 1);
num_elem_var.get_internal_symbol()->kind = SK_VARIABLE;
num_elem_var.set_type(loop_info[i].upper_bound.get_type().no_ref().get_unqualified_type());
num_elem_var.set_value(Nodecl::Div::make(
Nodecl::ParenthesizedExpression::make(
Nodecl::Add::make(
Nodecl::Minus::make(
Nodecl::ParenthesizedExpression::make(
loop_info[i].upper_bound,
loop_info[i].upper_bound.get_type()),
Nodecl::ParenthesizedExpression::make(
loop_info[i].lower_bound,
loop_info[i].lower_bound.get_type()),
loop_info[i].lower_bound.get_type()),
step_var.make_nodecl( true),
step_var.get_type()),
step_var.get_type()),
step_var.make_nodecl( true),
step_var.get_type()));
omp_capture_symbols.insert(num_elem_var);
Nodecl::NodeclBase num_elem_update;
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
num_elem_update = Nodecl::ExpressionStatement::make(
Nodecl::Assignment::make(
num_elem_var.make_nodecl( true),
Nodecl::ConditionalExpression::make(
Nodecl::LowerThan::make(
num_elem_var.make_nodecl( true),
const_value_to_nodecl(const_value_get_signed_int(0)),
num_elem_var.get_type()),
const_value_to_nodecl(const_value_get_signed_int(0)),
num_elem_var.make_nodecl( true),
num_elem_var.get_type()),
num_elem_var.get_type().get_lvalue_reference_to()));
}
else
{
num_elem_update = Nodecl::IfElseStatement::make(
Nodecl::LowerThan::make(
num_elem_var.make_nodecl( true),
const_value_to_nodecl(const_value_get_signed_int(0)),
num_elem_var.get_type()),
Nodecl::List::make(
Nodecl::ExpressionStatement::make(
Nodecl::Assignment::make(
num_elem_var.make_nodecl( true),
const_value_to_nodecl(const_value_get_signed_int(0)),
num_elem_var.get_type()))),
Nodecl::NodeclBase::null());
}
std::stringstream rounded_size_ss;
rounded_size_ss << "collapse_" << i << "_rounded_size";
TL::Symbol rounded_size_var = collapse_scope.new_symbol(rounded_size_ss.str());
symbol_entity_specs_set_is_user_declared(rounded_size_var.get_internal_symbol(), 1);
rounded_size_var.get_internal_symbol()->kind = SK_VARIABLE;
rounded_size_var.set_type(num_elem_var.get_type());
rounded_size_var.set_value(Nodecl::Mul::make(
num_elem_var.make_nodecl( true),
step_var.make_nodecl( true),
num_elem_var.get_type()));
omp_capture_symbols.insert(rounded_size_var);
collapse_statements.append(Nodecl::ObjectInit::make(step_var));
collapse_statements.append(Nodecl::ObjectInit::make(num_elem_var));
collapse_statements.append(num_elem_update);
collapse_statements.append(Nodecl::ObjectInit::make(rounded_size_var));
Nodecl::NodeclBase current_element_number = Nodecl::Conversion::make(
Nodecl::ParenthesizedExpression::make(
Nodecl::Div::make(
collapse_induction_var.make_nodecl( true),
Nodecl::ParenthesizedExpression::make(
num_elem_in_nested_loops,
num_elem_in_nested_loops.get_type()),
num_elem_in_nested_loops.get_type()),
num_elem_in_nested_loops.get_type()),
TL::Type::get_long_long_int_type());
current_element_number.set_text("C");
Nodecl::NodeclBase induction_var_expr = Nodecl::Conversion::make(
Nodecl::Add::make(
loop_info[i].lower_bound.shallow_copy(),
Nodecl::Mod::make(
Nodecl::ParenthesizedExpression::make(
Nodecl::Mul::make(
current_element_number,
step_var.make_nodecl( true),
step_var.get_type()),
step_var.get_type()),
rounded_size_var.make_nodecl( true),
rounded_size_var.get_type()),
loop_info[i].lower_bound.get_type()),
loop_info[i].induction_var.get_type());
induction_var_expr.set_text("C");
Nodecl::NodeclBase new_stmt;
if (collapse_scope.scope_is_enclosed_by(loop_info[i].induction_var.get_scope()))
{
new_stmt = Nodecl::ExpressionStatement::make(
Nodecl::Assignment::make(
loop_info[i].induction_var.make_nodecl( true),
induction_var_expr,
loop_info[i].induction_var.get_type().no_ref().get_lvalue_reference_to()));
omp_capture_symbols.insert(loop_info[i].induction_var);
}
else
{
TL::Symbol induction_var = loop_statements_scope.new_symbol("collapse_" + loop_info[i].induction_var.get_name());
symbol_entity_specs_set_is_user_declared(induction_var.get_internal_symbol(), 1);
induction_var.get_internal_symbol()->kind = SK_VARIABLE;
induction_var.set_type(loop_info[i].induction_var.get_type());
induction_var.set_value(induction_var_expr);
induction_var_map.add_map(loop_info[i].induction_var, induction_var);
new_stmt = Nodecl::ObjectInit::make(induction_var);
}
new_loop_statements.prepend(new_stmt);
num_elem_in_nested_loops = Nodecl::Mul::make(
num_elem_var.make_nodecl( true),
Nodecl::ParenthesizedExpression::make(
num_elem_in_nested_loops.shallow_copy(),
num_elem_in_nested_loops.get_type()),
num_elem_in_nested_loops.get_type());
}
condition_bound = num_elem_in_nested_loops;
Nodecl::Context new_context_innermost_loop =
Nodecl::Utils::deep_copy(
context_innermost_loop,
context_innermost_loop,
induction_var_map).as<Nodecl::Context>();
new_loop_statements.append(new_context_innermost_loop);
TL::Scope scope_innermost_loop = context_innermost_loop.retrieve_context();
scope_innermost_loop.get_decl_context()->current_scope->contained_in =
loop_statements_scope.get_decl_context()->current_scope;
}
void compute_loop_information(
Nodecl::NodeclBase node,
int collapse_factor,
TL::ObjectList<LoopInfo>& loop_info,
Nodecl::Context& context_innermost_loop,
Nodecl::List& post_trasformation_stmts)
{
for (int i = 0; i < collapse_factor; ++i)
{
TL::ForStatement for_stmt(node.as<Nodecl::ForStatement>());
LoopInfo info;
info.induction_var = for_stmt.get_induction_variable();
info.lower_bound = for_stmt.get_lower_bound();
info.upper_bound = for_stmt.get_upper_bound();
info.step = for_stmt.get_step();
loop_info.append(info);
context_innermost_loop =
for_stmt.get_statement().as<Nodecl::List>().front().as<Nodecl::Context>();
node = context_innermost_loop.get_in_context();
if (!for_stmt.induction_variable_in_separate_scope())
{
Nodecl::NodeclBase induction_variable =
for_stmt.get_induction_variable().make_nodecl( true);
Nodecl::NodeclBase expr =
HLT::Utils::compute_induction_variable_final_expr(for_stmt);
post_trasformation_stmts.append(
Nodecl::ExpressionStatement::make(
Nodecl::Assignment::make(
induction_variable,
expr,
induction_variable.get_type())));
}
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
node = node
.as<Nodecl::List>().front()
.as<Nodecl::CompoundStatement>().get_statements();
if (i < collapse_factor - 1)
{
node = node
.as<Nodecl::List>().front()
.as<Nodecl::Context>().get_in_context()
.as<Nodecl::List>().front();
}
}
else if (IS_FORTRAN_LANGUAGE && i < collapse_factor - 1)
{
node = node
.as<Nodecl::List>().front();
}
}
}
}
void LoopCollapse::collapse()
{
check_loop(_loop, _collapse_factor);
TL::ObjectList<LoopInfo> loop_info;
Nodecl::Context context_innermost_loop;
compute_loop_information(_loop, _collapse_factor, loop_info,
context_innermost_loop, _post_transformation_stmts);
TL::Scope collapse_scope = TL::Scope(
new_block_context(_pragma_context.get_decl_context()));
TL::Scope loop_control_scope = TL::Scope(
new_block_context(collapse_scope.get_decl_context()));
TL::Scope loop_statements_scope = TL::Scope(
new_block_context(loop_control_scope.get_decl_context()));
TL::Symbol induction_var = collapse_scope.new_symbol("collapse_it");
symbol_entity_specs_set_is_user_declared(induction_var.get_internal_symbol(), 1);
induction_var.get_internal_symbol()->kind = SK_VARIABLE;
induction_var.set_type(TL::Type::get_unsigned_long_long_int_type());
induction_var.set_value(const_value_to_nodecl(const_value_get_signed_int(0)));
Nodecl::List collapse_statements;
Nodecl::NodeclBase condition_bound;
Nodecl::List new_loop_statements;
compute_collapse_statements(
_collapse_factor, induction_var, context_innermost_loop,
loop_info, collapse_scope, loop_statements_scope,
collapse_statements, new_loop_statements, condition_bound, _omp_capture_symbols);
TL::Symbol condition_bound_var = collapse_scope.new_symbol("collapse_total_num_elements");
symbol_entity_specs_set_is_user_declared(condition_bound_var.get_internal_symbol(), 1);
condition_bound_var.get_internal_symbol()->kind = SK_VARIABLE;
condition_bound_var.set_type(TL::Type::get_unsigned_long_long_int_type());
collapse_statements.append(Nodecl::ObjectInit::make(condition_bound_var));
Nodecl::NodeclBase loop_control;
if (IS_C_LANGUAGE || IS_CXX_LANGUAGE)
{
condition_bound_var.set_value(condition_bound);
Nodecl::NodeclBase assignment = Nodecl::Assignment::make(
induction_var.make_nodecl( true),
const_value_to_nodecl(const_value_get_signed_int(0)),
induction_var.get_type().no_ref());
Nodecl::NodeclBase condition = Nodecl::LowerThan::make(
induction_var.make_nodecl( true),
condition_bound_var.make_nodecl( true),
TL::Type::get_bool_type());
Nodecl::NodeclBase step = Nodecl::Preincrement::make(
induction_var.make_nodecl( true),
induction_var.get_type());
loop_control = Nodecl::LoopControl::make(
Nodecl::List::make(assignment),
condition,
step);
}
else
{
condition_bound_var.set_value(Nodecl::Minus::make(
condition_bound,
const_value_to_nodecl(const_value_get_signed_int(1)),
condition_bound.get_type().no_ref()));
Nodecl::NodeclBase step =
const_value_to_nodecl(const_value_get_one( 4,  1));
loop_control = Nodecl::RangeLoopControl::make(
induction_var.make_nodecl( true),
const_value_to_nodecl(const_value_get_signed_int(0)),
condition_bound_var.make_nodecl( true),
step);
}
Nodecl::NodeclBase collapsed_loop = Nodecl::Context::make(
Nodecl::List::make(
Nodecl::ForStatement::make(
loop_control,
Nodecl::List::make(
Nodecl::Context::make(
Nodecl::List::make(
Nodecl::CompoundStatement::make(
new_loop_statements,
Nodecl::NodeclBase::null())),
loop_statements_scope)),
Nodecl::NodeclBase::null(),
_loop.get_locus())),
loop_control_scope.get_decl_context());
collapse_statements.append(collapsed_loop);
_transformation = Nodecl::Context::make(
Nodecl::List::make(
Nodecl::CompoundStatement::make(
collapse_statements,
Nodecl::NodeclBase::null())),
collapse_scope.get_decl_context());
}
}}
