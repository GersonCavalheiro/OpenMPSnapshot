#include "tl-omp-lowering-directive-environment.hpp"
#include "tl-omp-lowering-final-stmts-generator.hpp"
#include "tl-symbol-utils.hpp"
#include "cxx-cexpr.h"
namespace TL { namespace OpenMP { namespace Lowering {
Nodecl::NodeclBase FinalStmtsGenerator::generate_final_stmts(Nodecl::NodeclBase stmts)
{
class FinalStatementsPreVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
bool _ompss_mode;
const std::string & _in_final_fun_name;
int _num_task_related_pragmas;
TL::ObjectList<Nodecl::NodeclBase> _function_codes_to_be_duplicated;
TL::ObjectList<Nodecl::NodeclBase> _already_visited;
const Nodecl::Utils::SimpleSymbolMap& _function_translation_map;
public:
FinalStatementsPreVisitor(bool ompss_mode, const std::string &in_final_fun_name,
const Nodecl::Utils::SimpleSymbolMap& function_tranlation_map)
:
_ompss_mode(ompss_mode),
_in_final_fun_name(in_final_fun_name),
_num_task_related_pragmas(0),
_function_codes_to_be_duplicated(),
_already_visited(),
_function_translation_map(function_tranlation_map) { }
void visit(const Nodecl::OpenMP::Taskwait& taskwait)
{
++_num_task_related_pragmas;
}
void visit(const Nodecl::OpenMP::Task& task)
{
++_num_task_related_pragmas;
walk(task.get_statements());
}
void visit(const Nodecl::OmpSs::TaskCall& task_call)
{
++_num_task_related_pragmas;
walk(task_call.get_call());
}
void visit(const Nodecl::OpenMP::Taskloop& taskloop)
{
++_num_task_related_pragmas;
walk(taskloop.get_loop());
}
void visit(const Nodecl::OmpSs::TaskExpression& task_expr)
{
++_num_task_related_pragmas;
walk(task_expr.get_sequential_code());
}
void visit(const Nodecl::OmpSs::TaskWorksharing &loop)
{
++_num_task_related_pragmas;
walk(loop.get_loop());
}
void visit(const Nodecl::OmpSs::TaskloopWorksharing &loop)
{
++_num_task_related_pragmas;
walk(loop.get_loop());
}
void visit(const Nodecl::ObjectInit& object_init)
{
TL::Symbol sym = object_init.get_symbol();
Nodecl::NodeclBase value = sym.get_value();
if (!value.is_null())
walk(value);
}
void visit(const Nodecl::FunctionCall &function_call)
{
Nodecl::NodeclBase called = function_call.get_called();
if (!called.is<Nodecl::Symbol>())
return;
TL::Symbol called_sym = called.as<Nodecl::Symbol>().get_symbol();
if (called_sym.get_name() == _in_final_fun_name)
{
++_num_task_related_pragmas;
return;
}
Nodecl::NodeclBase function_code = called_sym.get_function_code();
if (function_code.is_null())
return;
if (_already_visited.contains(function_code))
return;
const std::map<TL::Symbol, TL::Symbol>* map =
_function_translation_map.get_simple_symbol_map();
if (map->find(called_sym) != map->end())
return;
_already_visited.append(function_code);
int old_num_tasks_detected = _num_task_related_pragmas;
walk(function_code);
if (old_num_tasks_detected != _num_task_related_pragmas)
_function_codes_to_be_duplicated.append(function_code);
}
TL::ObjectList<Nodecl::NodeclBase>& get_function_codes_to_be_duplicated()
{
return _function_codes_to_be_duplicated;
}
};
class FinalStatementsGenerator : public Nodecl::ExhaustiveVisitor<void>
{
private:
bool _ompss_mode;
const std::string &_in_final_fun_name;
Nodecl::NodeclBase _enclosing_function_code;
Nodecl::Utils::SimpleSymbolMap& _function_translation_map;
const TL::ObjectList<Nodecl::NodeclBase>& _function_codes_to_be_duplicated;
public:
FinalStatementsGenerator(
bool ompss_mode,
const std::string in_final_fun_name,
Nodecl::NodeclBase enclosing_function_code,
Nodecl::Utils::SimpleSymbolMap& function_tranlation_map,
const TL::ObjectList<Nodecl::NodeclBase>& function_codes_to_be_duplicated)
:
_ompss_mode(ompss_mode),
_in_final_fun_name(in_final_fun_name),
_enclosing_function_code(enclosing_function_code),
_function_translation_map(function_tranlation_map),
_function_codes_to_be_duplicated(function_codes_to_be_duplicated) { }
void visit(const Nodecl::OpenMP::Taskwait& taskwait)
{
Nodecl::Utils::remove_from_enclosing_list(taskwait);
}
void visit(const Nodecl::OpenMP::Task& task)
{
task.replace(task.get_statements());
walk(task);
}
void visit(const Nodecl::OmpSs::TaskCall& task_call)
{
DirectiveEnvironment _env = task_call.get_environment();
ERROR_CONDITION(_env.device_names.size() > 1, "Unexpected device clause list\n", 0);
bool is_cuda_task = (*_env.device_names.begin() == "cuda");
if (!is_cuda_task) {
task_call.replace(task_call.get_call());
walk(task_call);
} else {
walk(task_call.get_call());
}
}
void visit(const Nodecl::OpenMP::Taskloop& taskloop)
{
taskloop.replace(taskloop.get_loop());
walk(taskloop);
}
void visit(const Nodecl::OmpSs::TaskWorksharing &loop)
{
loop.replace(loop.get_loop());
walk(loop);
}
void visit(const Nodecl::OmpSs::TaskloopWorksharing &loop)
{
loop.replace(loop.get_loop());
walk(loop);
}
void visit(const Nodecl::OmpSs::TaskExpression& task_expr)
{
Nodecl::NodeclBase seq_code = task_expr.get_sequential_code();
ERROR_CONDITION(!seq_code.is<Nodecl::ExpressionStatement>(), "Unreachable code\n", 0);
task_expr.replace(seq_code.as<Nodecl::ExpressionStatement>().get_nest());
walk(task_expr);
}
void visit(const Nodecl::ObjectInit& object_init)
{
TL::Symbol sym = object_init.get_symbol();
Nodecl::NodeclBase value = sym.get_value();
if (!value.is_null())
walk(value);
}
void visit(const Nodecl::FunctionCall& function_call)
{
Nodecl::NodeclBase called = function_call.get_called();
if (!called.is<Nodecl::Symbol>())
return;
TL::Symbol called_sym = called.as<Nodecl::Symbol>().get_symbol();
if (called_sym.get_name() == _in_final_fun_name)
{
nodecl_t true_expr;
if (IS_FORTRAN_LANGUAGE)
{
true_expr = nodecl_make_boolean_literal(
get_bool_type(),
const_value_get_one(type_get_size(get_bool_type()), 0),
function_call.get_locus());
}
else
true_expr = const_value_to_nodecl(const_value_get_signed_int(1));
function_call.replace(true_expr);
return;
}
Nodecl::NodeclBase function_code = called_sym.get_function_code();
if (!function_code.is_null())
{
const std::map<TL::Symbol, TL::Symbol>* map =
_function_translation_map.get_simple_symbol_map();
bool has_been_duplicated = map->find(called_sym) != map->end();
if (
_function_codes_to_be_duplicated.contains(function_code)
&& !has_been_duplicated)
{
TL::Symbol new_function_sym = SymbolUtils::new_function_symbol_for_deep_copy(
called_sym,
called_sym.get_name() + "_mcc_serial");
has_been_duplicated = true;
_function_translation_map.add_map(called_sym, new_function_sym);
Nodecl::NodeclBase new_function_code = Nodecl::Utils::deep_copy(
function_code,
called_sym.get_scope(),
_function_translation_map);
if (called_sym.is_member())
{
::class_type_add_member(
symbol_entity_specs_get_class_type(
new_function_sym.get_internal_symbol()),
new_function_sym.get_internal_symbol(),
new_function_sym.get_internal_symbol()->decl_context,
1);
}
else
{
CXX_LANGUAGE()
{
Nodecl::NodeclBase nodecl_decl = Nodecl::CxxDecl::make(
nodecl_null(),
new_function_sym,
function_call.get_locus());
Nodecl::Utils::prepend_items_before(_enclosing_function_code, nodecl_decl);
}
}
Nodecl::Utils::prepend_items_before(function_code, new_function_code);
Nodecl::NodeclBase old_enclosing_funct_code = _enclosing_function_code;
_enclosing_function_code = new_function_code;
walk(new_function_code);
_enclosing_function_code = old_enclosing_funct_code;
}
if (has_been_duplicated)
{
Nodecl::NodeclBase new_function_call = Nodecl::Utils::deep_copy(
function_call,
function_call,
_function_translation_map);
function_call.replace(new_function_call);
}
}
}
};
Nodecl::NodeclBase new_stmts = Nodecl::Utils::deep_copy(stmts, stmts.retrieve_context());
FinalStatementsPreVisitor pre_visitor(_ompss_mode, _in_final_fun_name, _function_translation_map);
pre_visitor.walk(new_stmts);
TL::Symbol enclosing_funct_sym = Nodecl::Utils::get_enclosing_function(stmts);
Nodecl::NodeclBase enclosing_funct_code = enclosing_funct_sym.get_function_code();
FinalStatementsGenerator generator(
_ompss_mode,
_in_final_fun_name,
enclosing_funct_code,
_function_translation_map,
pre_visitor.get_function_codes_to_be_duplicated());
generator.walk(new_stmts);
return new_stmts;
}
FinalStmtsGenerator::FinalStmtsGenerator(bool ompss_mode, const std::string &in_final_fun_name)
: _ompss_mode(ompss_mode),
_in_final_fun_name(in_final_fun_name),
_final_stmts_map(),
_function_translation_map() { }
void FinalStmtsGenerator::visit(const Nodecl::OpenMP::Task& task)
{
walk(task.get_statements());
Nodecl::NodeclBase final_stmts = generate_final_stmts(task.get_statements());
_final_stmts_map.insert(std::make_pair(task, final_stmts));
}
void FinalStmtsGenerator::visit(const Nodecl::OmpSs::TaskCall& task_call)
{
walk(task_call.get_call());
Nodecl::NodeclBase final_stmts = generate_final_stmts(task_call.get_call());
_final_stmts_map.insert(std::make_pair(task_call, final_stmts));
}
void FinalStmtsGenerator::visit(const Nodecl::OpenMP::Taskloop& node)
{
walk(node.get_loop());
Nodecl::NodeclBase final_stmts = generate_final_stmts(node.get_loop());
_final_stmts_map.insert(std::make_pair(node, final_stmts));
}
void FinalStmtsGenerator::visit(const Nodecl::OmpSs::TaskWorksharing &node)
{
walk(node.get_loop());
Nodecl::NodeclBase final_stmts = generate_final_stmts(node.get_loop());
_final_stmts_map.insert(std::make_pair(node, final_stmts));
}
void FinalStmtsGenerator::visit(const Nodecl::OmpSs::TaskloopWorksharing &node)
{
walk(node.get_loop());
Nodecl::NodeclBase final_stmts = generate_final_stmts(node.get_loop());
_final_stmts_map.insert(std::make_pair(node, final_stmts));
}
void FinalStmtsGenerator::visit(const Nodecl::OmpSs::TaskExpression& task_expr)
{
walk(task_expr.get_sequential_code());
Nodecl::NodeclBase final_stmts = generate_final_stmts(task_expr.get_sequential_code());
_final_stmts_map.insert(std::make_pair(task_expr, final_stmts));
}
std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>& FinalStmtsGenerator::get_final_stmts()
{
return _final_stmts_map;
}
}}}
