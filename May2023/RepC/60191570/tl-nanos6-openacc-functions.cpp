#include "tl-nanos6-openacc-functions.hpp"
#include "tl-nodecl-visitor.hpp"
#include "tl-omp-core.hpp"
#include "cxx-cexpr.h"
namespace TL
{
namespace Nanos6
{
OpenACCTasks::OpenACCTasks()
{
set_phase_name("Nanos6/OpenACC specific pass");
set_phase_description(
"This pass makes sure all the device specific information that Nanos6 "
"has is properly passed to OpenACC directives.");
}
void OpenACCTasks::append_async_parameter(TL::Symbol &sym)
{
TL::Symbol new_param;	
TL::ObjectList<TL::Symbol> current_params = sym.get_related_symbols();
TL::Type func_type = sym.get_type();
TL::Type ret_type = func_type.returns();
TL::ObjectList<TL::Type> param_types = func_type.parameters();
TL::Scope sc = current_params.begin()->get_scope();
new_param = sc.new_symbol("nanos6_mcxx_async_queue");
new_param.get_internal_symbol()->kind = SK_VARIABLE;
new_param.set_type(TL::Type::get_int_type());	
symbol_entity_specs_set_is_user_declared(new_param.get_internal_symbol(), 1);
param_types.append(new_param.get_type());
func_type = ret_type.get_function_returning(param_types, param_types);
sym.set_type(func_type);
current_params.append(new_param);
sym.set_related_symbols(current_params);
}
class UnknownPragmaVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
TL::OmpSs::FunctionTaskSet &ompss_task_functions;
TL::ObjectList<TL::Symbol> statements;
std::string append_async(std::string pragma_str)
{
const std::string append_str = " async(nanos6_mcxx_async_queue)";
std::string ret = pragma_str.substr(pragma_str.find("acc"));
size_t length = ret.find("\n");
ret = ret.erase(length, 1);	
ret += append_str;
return ret;
}
public:
UnknownPragmaVisitor(TL::OmpSs::FunctionTaskSet &ompss_task_functions_)
: ompss_task_functions(ompss_task_functions_)
{
}
virtual void visit(const Nodecl::UnknownPragma &node)
{
if (node.prettyprint().find(" acc parallel") != std::string::npos ||
node.prettyprint().find(" acc kernels") != std::string::npos)
{
ERROR_CONDITION(
node.prettyprint().find("async") != std::string::npos,
"OpenACC async clause already present, please remove from user code", 0);
std::string acc_pragma_str = node.prettyprint();
acc_pragma_str = append_async(acc_pragma_str);
const_cast<Nodecl::UnknownPragma&>(node).set_text(acc_pragma_str);
}
}
};
class FunctionDefinitionsVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
TL::OmpSs::FunctionTaskSet &ompss_task_functions;
TL::ObjectList<TL::Symbol> openacc_functions;
public:
FunctionDefinitionsVisitor(TL::OmpSs::FunctionTaskSet &ompss_task_functions_)
: ompss_task_functions(ompss_task_functions_)
{
}
virtual void visit(const Nodecl::FunctionCode &node)
{
TL::Symbol sym = node.get_symbol();
if (!ompss_task_functions.is_function_task(sym))
return;
TL::OmpSs::FunctionTaskInfo &task_info
= ompss_task_functions.get_function_task(sym);
TL::OmpSs::TargetInfo &target_info = task_info.get_target_info();
TL::ObjectList<std::string> devices = target_info.get_device_list();
if (devices.contains("openacc"))
{
if (devices.size() == 1)
{
openacc_functions.insert(sym);
}
else
{
error_printf_at(
node.get_locus(),
"OpenACC function task is using more than one device\n");
}
}
}
TL::ObjectList<TL::Symbol> get_openacc_functions_definitions() const
{
return openacc_functions;
}
};
class FunctionCodeVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
TL::OmpSs::FunctionTaskSet &ompss_task_functions;
public:
FunctionCodeVisitor(TL::OmpSs::FunctionTaskSet &ompss_task_functions_)
: ompss_task_functions(ompss_task_functions_)
{
}
virtual void visit(const Nodecl::FunctionCode &node)
{
TL::Symbol sym = node.get_symbol();
if (!ompss_task_functions.is_function_task(sym))
return;
TL::Symbol async = sym.get_related_symbols().begin()->
get_scope().get_symbol_from_name("nanos6_mcxx_async_queue");
ERROR_CONDITION(!async.is_valid(),
"async queue symbol not found\n", 0);
Nodecl::Context context = node.get_statements().as<Nodecl::Context>();
Nodecl::List statements = context.get_in_context().as<Nodecl::List>();
Nodecl::CompoundStatement cm_statement = statements.front().as<Nodecl::CompoundStatement>();
Nodecl::NodeclBase pragma_acc_wait = Nodecl::UnknownPragma::make("acc wait(0)");
Nodecl::NodeclBase if_async_zero = Nodecl::IfElseStatement::make(
Nodecl::Equal::make(
async.make_nodecl(true),	
const_value_to_nodecl_with_basic_type(	
const_value_get_signed_int(0),		
get_size_t_type()),
get_bool_type()),
Nodecl::List::make(		
Nodecl::CompoundStatement::make( 
Nodecl::List::make(
pragma_acc_wait,				
Nodecl::EmptyStatement::make()),
Nodecl::NodeclBase::null())),		
Nodecl::NodeclBase::null());
Nodecl::List stmt_list = cm_statement.get_statements().as<Nodecl::List>();
ERROR_CONDITION(stmt_list.empty(), "Statement list appears empty\n", 0);
stmt_list.append(if_async_zero);
}
};
class FunctionCallsVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
TL::OmpSs::FunctionTaskSet &ompss_task_functions;
public:
FunctionCallsVisitor(TL::OmpSs::FunctionTaskSet &ompss_task_functions_)
: ompss_task_functions(ompss_task_functions_)
{
}
virtual void visit(const Nodecl::FunctionCall &node)
{
Nodecl::NodeclBase called = node.get_called();
if (!called.is<Nodecl::Symbol>())
return;
TL::Symbol sym = called.get_symbol();
if (!ompss_task_functions.is_function_task(sym))
return;
TL::OmpSs::FunctionTaskInfo &task_info
= ompss_task_functions.get_function_task(sym);
TL::OmpSs::TargetInfo &target_info = task_info.get_target_info();
TL::ObjectList<std::string> devices = target_info.get_device_list();
if (devices.contains("openacc"))
{
Nodecl::List arguments = node.get_arguments().as<Nodecl::List>();
TL::Scope sc = node.retrieve_context();
const std::string new_arg_name = "nanos6_mcxx_async";
TL::Symbol async_symbol = sc.get_symbol_from_name(new_arg_name);
if (async_symbol.is_valid())
{
arguments.append(Nodecl::Conversion::make(
async_symbol.make_nodecl( true, node.get_locus()),
TL::Type::get_int_type(),
node.get_locus()));
}
else
{
arguments.append(Nodecl::Conversion::make(
const_value_to_nodecl(const_value_get_zero(4, 1)),
TL::Type::get_int_type(),
node.get_locus()));
}
}
}
};
void OpenACCTasks::run(DTO &dto)
{
Nodecl::NodeclBase translation_unit
= *std::static_pointer_cast<Nodecl::NodeclBase>(dto["nodecl"]);
std::shared_ptr<TL::OmpSs::FunctionTaskSet> ompss_task_functions
= std::static_pointer_cast<TL::OmpSs::FunctionTaskSet>(
dto["openmp_task_info"]);
ERROR_CONDITION(
!ompss_task_functions, "OmpSs Task Functions not in the DTO", 0);
FunctionDefinitionsVisitor functions_definition_visitor(
*ompss_task_functions);
functions_definition_visitor.walk(translation_unit);
TL::ObjectList<TL::Symbol> acc_functions =
functions_definition_visitor.get_openacc_functions_definitions();
std::map<TL::Symbol, TL::OmpSs::FunctionTaskInfo> task_map
= ompss_task_functions->get_map();
for (auto p : task_map)
{
TL::Symbol sym = p.first;
TL::OmpSs::FunctionTaskInfo &task_info = p.second;
TL::OmpSs::TargetInfo &target_info = task_info.get_target_info();
TL::ObjectList<std::string> devices = target_info.get_device_list();
if (devices.contains("openacc"))
{
if (devices.size() == 1)
{
append_async_parameter(sym);
}
else
{
error_printf_at(
NULL,
"OpenACC function task is using more than one device\n");
}
}
}
FunctionCallsVisitor function_calls_visitor(
*ompss_task_functions);
function_calls_visitor.walk(translation_unit);
UnknownPragmaVisitor unknown_pragma_visitor(
*ompss_task_functions);
FunctionCodeVisitor code_visitor(
*ompss_task_functions);
for (auto f : acc_functions)
{
unknown_pragma_visitor.walk(f.get_function_code());
code_visitor.walk(f.get_function_code());
}
}
} 
} 
EXPORT_PHASE(TL::Nanos6::OpenACCTasks)
