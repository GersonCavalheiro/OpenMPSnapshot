#include "tl-nanos6-lower.hpp"
#include "tl-nanos6-task-properties.hpp"
#include "tl-nanos6-fortran-support.hpp"
#include "tl-nanos6-interface.hpp"
#include "tl-nanos6-support.hpp"
#include "tl-counters.hpp"
#include "tl-source.hpp"
#include "cxx-exprtype.h"
namespace TL { namespace Nanos6 {
void Lower::visit(const Nodecl::OpenMP::Taskloop& node)
{
Interface::family_must_be_at_least("nanos6_instantiation_api", 2, "to support taskloop");
Interface::family_must_be_at_least("nanos6_loop_api", 2, "to support taskloop");
Nodecl::OpenMP::Taskloop taskloop = node;
walk(taskloop.get_loop());
Nodecl::NodeclBase serial_stmts;
if (!_phase->_final_clause_transformation_disabled)
{
std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>::iterator it = _final_stmts_map.find(taskloop);
ERROR_CONDITION(it == _final_stmts_map.end(), "Invalid serial statements", 0);
serial_stmts = it->second;
}
lower_taskloop(taskloop, serial_stmts);
}
namespace {
void create_final_if_else_statement(Nodecl::OpenMP::Taskloop& node, Nodecl::NodeclBase& serial_stmts_placeholder)
{
Nodecl::NodeclBase stmts = node.get_loop();
TL::Symbol nanos_in_final_sym = get_nanos6_function_symbol("nanos6_in_final");
Nodecl::NodeclBase call_to_nanos_in_final = Nodecl::FunctionCall::make(
nanos_in_final_sym.make_nodecl( true, node.get_locus()),
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null(),
TL::Type::get_int_type());
Nodecl::OpenMP::Taskloop new_task = Nodecl::OpenMP::Taskloop::make(node.get_environment(), stmts, node.get_locus());
Scope sc = node.retrieve_context();
Scope not_final_context = new_block_context(sc.get_decl_context());
Nodecl::NodeclBase not_final_compound_stmt = Nodecl::Context::make(
Nodecl::List::make(
Nodecl::CompoundStatement::make(
Nodecl::List::make(new_task),
Nodecl::NodeclBase::null(),
node.get_locus())),
not_final_context,
node.get_locus());
serial_stmts_placeholder = Nodecl::EmptyStatement::make();
Nodecl::NodeclBase if_in_final = Nodecl::IfElseStatement::make(
Nodecl::Different::make(
call_to_nanos_in_final,
const_value_to_nodecl_with_basic_type(
const_value_get_signed_int(0),
get_size_t_type()),
get_bool_type()),
Nodecl::List::make(serial_stmts_placeholder),
Nodecl::List::make(not_final_compound_stmt)
);
node.replace(if_in_final);
node = new_task;
}
void handle_task_transformation(const Nodecl::OpenMP::Taskloop& node, TaskProperties& task_properties)
{
Nodecl::NodeclBase args_size;
TL::Type data_env_struct;
bool requires_initialization;
task_properties.create_environment_structure(
data_env_struct,
args_size,
requires_initialization);
TL::Symbol task_invocation_info;
task_properties.create_task_invocation_info(
task_invocation_info);
TL::Symbol implementations;
task_properties.create_task_implementations_info(
implementations);
TL::Symbol task_info;
task_properties.create_task_info(
implementations,
task_info);
TL::Scope sc = node.retrieve_context();
TL::Symbol args;
{
TL::Counter &counter = TL::CounterManager::get_counter("nanos6-taskloop-args");
std::stringstream ss;
ss << "nanos_data_env_" << (int)counter;
counter++;
args = sc.new_symbol(ss.str());
args.get_internal_symbol()->kind = SK_VARIABLE;
args.set_type(data_env_struct.get_pointer_to());
symbol_entity_specs_set_is_user_declared(args.get_internal_symbol(), 1);
}
TL::Symbol task_ptr;
{
TL::Counter &counter = TL::CounterManager::get_counter("nanos6-taskloop-ptr");
std::stringstream ss;
ss << "nanos_task_ptr_" << (int)counter;
counter++;
task_ptr = sc.new_symbol(ss.str());
task_ptr.get_internal_symbol()->kind = SK_VARIABLE;
task_ptr.set_type(TL::Type::get_void_type().get_pointer_to());
symbol_entity_specs_set_is_user_declared(task_ptr.get_internal_symbol(), 1);
}
Nodecl::List new_stmts;
if (IS_FORTRAN_LANGUAGE)
{
TL::Symbol intrinsic_null = get_fortran_intrinsic_symbol<0>("null", Nodecl::List(),  0);
new_stmts.append(
Nodecl::ExpressionStatement::make(
Nodecl::Assignment::make(
args.make_nodecl(true),
Nodecl::FunctionCall::make(
intrinsic_null.make_nodecl( true),
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null(),
intrinsic_null.get_type().returns(),
node.get_locus()),
args.get_type())));
}
{
if (IS_CXX_LANGUAGE)
{
new_stmts.append(Nodecl::CxxDef::make(Nodecl::NodeclBase::null(), args));
new_stmts.append(Nodecl::CxxDef::make(Nodecl::NodeclBase::null(), task_ptr));
}
TL::Symbol nanos_create_task_sym;
if (Interface::family_is_at_least("nanos6_loop_api", 3))
{
nanos_create_task_sym = get_nanos6_function_symbol("nanos6_create_loop");
}
else
{
nanos_create_task_sym = get_nanos6_function_symbol("nanos6_create_task");
}
Nodecl::List create_task_args;
Nodecl::NodeclBase task_info_ptr =
Nodecl::Reference::make(
task_info.make_nodecl(
true,
node.get_locus()),
task_info.get_type().get_pointer_to(),
node.get_locus());
create_task_args.append(task_info_ptr);
Nodecl::NodeclBase task_invocation_info_ptr =
Nodecl::Reference::make(
task_invocation_info.make_nodecl(
true,
node.get_locus()),
task_invocation_info.get_type().get_pointer_to(),
node.get_locus());
create_task_args.append(task_invocation_info_ptr);
if (Interface::family_is_at_least("nanos6_instantiation_api", 5))
{
if (IS_FORTRAN_LANGUAGE)
{
create_task_args.append(
const_value_to_nodecl(const_value_make_string_null_ended("", 1)));
}
else
{
create_task_args.append(
const_value_to_nodecl(const_value_get_signed_int(0)));
}
}
create_task_args.append(args_size);
Nodecl::NodeclBase cast;
Nodecl::NodeclBase args_ptr_out =
cast = Nodecl::Conversion::make(
Nodecl::Reference::make(
args.make_nodecl(
true,
node.get_locus()),
args.get_type().get_pointer_to(),
node.get_locus()),
TL::Type::get_void_type().get_pointer_to().get_pointer_to(),
node.get_locus());
cast.set_text("C");
create_task_args.append(args_ptr_out);
Nodecl::NodeclBase task_ptr_out =
Nodecl::Reference::make(
task_ptr.make_nodecl(
true,
node.get_locus()),
task_ptr.get_type().get_pointer_to(),
node.get_locus());
create_task_args.append(task_ptr_out);
{
TL::Symbol task_flags;
{
TL::Counter &counter = TL::CounterManager::get_counter("nanos6-taskloop-flags");
std::stringstream ss;
ss << "task_flags_" << (int)counter;
counter++;
task_flags = sc.new_symbol(ss.str());
task_flags.get_internal_symbol()->kind = SK_VARIABLE;
task_flags.get_internal_symbol()->type_information = TL::Type::get_size_t_type().get_internal_type();
symbol_entity_specs_set_is_user_declared(task_flags.get_internal_symbol(), 1);
}
if (IS_CXX_LANGUAGE)
new_stmts.append(Nodecl::CxxDef::make(Nodecl::NodeclBase::null(), task_flags));
Nodecl::NodeclBase task_flags_stmts;
task_properties.compute_task_flags(task_flags, task_flags_stmts);
new_stmts.append(task_flags_stmts);
create_task_args.append(task_flags.make_nodecl( true));
}
{
TL::Symbol num_deps;
{
TL::Counter &counter = TL::CounterManager::get_counter("nanos6-num-dependences");
std::stringstream ss;
ss << "nanos_num_deps_" << (int)counter;
counter++;
num_deps = sc.new_symbol(ss.str());
num_deps.get_internal_symbol()->kind = SK_VARIABLE;
num_deps.set_type(TL::Type::get_size_t_type());
symbol_entity_specs_set_is_user_declared(num_deps.get_internal_symbol(), 1);
}
Nodecl::NodeclBase compute_num_deps_stmts;
task_properties.compute_number_of_dependences(num_deps, sc,  compute_num_deps_stmts);
new_stmts.append(Nodecl::ObjectInit::make(num_deps));
if (IS_CXX_LANGUAGE)
new_stmts.append(Nodecl::CxxDef::make(Nodecl::NodeclBase::null(), num_deps));
new_stmts.append(compute_num_deps_stmts);
create_task_args.append(num_deps.make_nodecl( true));
}
if (Interface::family_is_at_least("nanos6_loop_api", 3))
{
Nodecl::NodeclBase lower_bound = task_properties.get_lower_bound().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
lower_bound = Nodecl::Conversion::make(lower_bound, TL::Type::get_size_t_type());
create_task_args.append(lower_bound);
Nodecl::NodeclBase upper_bound = task_properties.get_upper_bound().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
upper_bound = Nodecl::Conversion::make(upper_bound, TL::Type::get_size_t_type());
create_task_args.append(upper_bound);
Nodecl::NodeclBase grainsize = task_properties.get_grainsize().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
grainsize = Nodecl::Conversion::make(grainsize, TL::Type::get_size_t_type());
create_task_args.append(grainsize);
Nodecl::NodeclBase chunksize = task_properties.get_chunksize().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
chunksize = Nodecl::Conversion::make(chunksize, TL::Type::get_size_t_type());
create_task_args.append(chunksize);
}
if (IS_FORTRAN_LANGUAGE)
{
Nodecl::NodeclBase allocate_stmt = Nodecl::FortranAllocateStatement::make(
Nodecl::List::make(args.make_nodecl( true)),
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null());
new_stmts.append(allocate_stmt);
}
Nodecl::NodeclBase call_to_nanos_create_task =
Nodecl::ExpressionStatement::make(
Nodecl::FunctionCall::make(
nanos_create_task_sym.make_nodecl( true, node.get_locus()),
create_task_args,
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null(),
TL::Type::get_void_type(),
node.get_locus()),
node.get_locus());
new_stmts.append(call_to_nanos_create_task);
}
{
Nodecl::NodeclBase capture_env;
task_properties.capture_environment(
args,
sc,
capture_env);
new_stmts.append(capture_env);
}
if (!Interface::family_is_at_least("nanos6_loop_api", 3))
{
TL::Symbol nanos_register_loop_sym = get_nanos6_register_loop_bounds_function();
Nodecl::NodeclBase stmt = node.get_loop();
ERROR_CONDITION(!stmt.is<Nodecl::Context>(), "Unexpected node\n", 0);
stmt = stmt.as<Nodecl::Context>().get_in_context().as<Nodecl::List>().front();
ERROR_CONDITION(!stmt.is<Nodecl::ForStatement>(), "Unexpected node\n", 0);
TL::ObjectList<TL::Symbol> params = nanos_register_loop_sym.get_related_symbols();
Nodecl::List reg_loop_args;
reg_loop_args.append(task_ptr.make_nodecl(true));
Nodecl::NodeclBase lower_bound = task_properties.get_lower_bound().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
lower_bound = Nodecl::Conversion::make(lower_bound, params[1].get_type());
reg_loop_args.append(lower_bound);
Nodecl::NodeclBase upper_bound = task_properties.get_upper_bound().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
upper_bound = Nodecl::Conversion::make(upper_bound, params[2].get_type());
reg_loop_args.append(upper_bound);
Nodecl::NodeclBase grainsize = task_properties.get_grainsize().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
grainsize = Nodecl::Conversion::make(grainsize, params[3].get_type());
reg_loop_args.append(grainsize);
Nodecl::NodeclBase chunksize = task_properties.get_chunksize().shallow_copy();
if (IS_FORTRAN_LANGUAGE)
chunksize = Nodecl::Conversion::make(chunksize, params[4].get_type());
reg_loop_args.append(chunksize);
new_stmts.append(
Nodecl::ExpressionStatement::make(
Nodecl::FunctionCall::make(
nanos_register_loop_sym.make_nodecl(true),
reg_loop_args,
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null(),
TL::Type::get_void_type(),
node.get_locus()),
node.get_locus()));
}
{
TL::Symbol nanos_submit_task_sym = get_nanos6_function_symbol("nanos6_submit_task");
Nodecl::NodeclBase task_ptr_arg = task_ptr.make_nodecl( true);
task_ptr_arg = ::cxx_nodecl_make_conversion(
task_ptr_arg.get_internal_nodecl(),
task_ptr.get_type().no_ref().get_internal_type(),
TL::Scope::get_global_scope().get_decl_context(),
node.get_locus());
Nodecl::NodeclBase new_task =
Nodecl::ExpressionStatement::make(
Nodecl::FunctionCall::make(
nanos_submit_task_sym.make_nodecl( true),
Nodecl::List::make(task_ptr_arg),
Nodecl::NodeclBase::null(),
Nodecl::NodeclBase::null(),
TL::Type::get_void_type(),
node.get_locus()),
node.get_locus());
new_stmts.append(new_task);
}
node.replace(new_stmts);
}
}
void Lower::lower_taskloop(const Nodecl::OpenMP::Taskloop& node, const Nodecl::NodeclBase& serial_stmts)
{
ERROR_CONDITION(!_phase->_final_clause_transformation_disabled
&& serial_stmts.is_null(),
"Invalid serial statements for a taskloop", 0);
Nodecl::OpenMP::Taskloop taskloop = node;
Nodecl::NodeclBase final_stmts;
Nodecl::NodeclBase serial_stmts_placeholder;
if (!_phase->_final_clause_transformation_disabled)
{
Scope in_final_scope =
new_block_context(taskloop.retrieve_context().get_decl_context());
final_stmts = Nodecl::Context::make(
Nodecl::List::make(
Nodecl::CompoundStatement::make(
Nodecl::List::make(serial_stmts),
Nodecl::NodeclBase::null(),
taskloop.get_locus())),
in_final_scope,
taskloop.get_locus());
create_final_if_else_statement(taskloop, serial_stmts_placeholder);
}
TaskProperties task_properties(taskloop, final_stmts, _phase, this);
handle_task_transformation(taskloop, task_properties);
if (!_phase->_final_clause_transformation_disabled)
{
serial_stmts_placeholder.replace(final_stmts);
walk(final_stmts);
}
}
} }
