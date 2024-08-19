#ifndef TL_OMP_BASE_TASK_HPP
#define TL_OMP_BASE_TASK_HPP
#include"tl-nodecl-visitor.hpp"
#include "tl-omp-core.hpp"
#include "tl-omp-base.hpp"
namespace TL { namespace OmpSs {
typedef std::map<TL::Symbol, Nodecl::NodeclBase> sym_to_argument_expr_t;
class FunctionCallVisitor : public Nodecl::ExhaustiveVisitor<void>
{
private:
std::shared_ptr<FunctionTaskSet> _function_task_set;
typedef std::set<Symbol> module_function_tasks_set_t;
module_function_tasks_set_t _module_function_tasks;
const std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>& _funct_call_to_enclosing_stmt_map;
const std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>& _enclosing_stmt_to_original_stmt_map;
const std::map<Nodecl::NodeclBase, std::set<TL::Symbol> >& _enclosing_stmt_to_return_vars_map;
std::map<Nodecl::NodeclBase, TL::ObjectList<Nodecl::NodeclBase> > _enclosing_stmt_to_task_calls_map;
OpenMP::Base *_base;
bool _ignore_template_functions;
public:
FunctionCallVisitor(std::shared_ptr<FunctionTaskSet> function_task_set,
const std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>& funct_call_to_enclosing_stmt_map,
const std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>& enclosing_stmt_to_original_stmt_map,
const std::map<Nodecl::NodeclBase, std::set<TL::Symbol> >& enclosing_stmt_to_return_vars_map,
OpenMP::Base* base,
bool ignore_template_functions);
virtual void visit(const Nodecl::FunctionCode& node);
virtual void visit(const Nodecl::TemplateFunctionCode& node);
virtual void visit(const Nodecl::FunctionCall& call);
void build_all_needed_task_expressions();
private:
Nodecl::NodeclBase make_exec_environment(const Nodecl::FunctionCall &call,
TL::Symbol function_sym,
FunctionTaskInfo& function_task_info);
void fill_map_parameters_to_arguments(
TL::Symbol function,
Nodecl::List arguments,
sym_to_argument_expr_t& param_to_arg_expr);
FunctionTaskInfo instantiate_function_task_info(
const FunctionTaskInfo& function_task_info,
TL::Scope context_of_being_instantiated,
instantiation_symbol_map_t* instantiation_symbol_map);
Nodecl::NodeclBase instantiate_exec_env(Nodecl::NodeclBase exec_env, Nodecl::FunctionCall call);
Nodecl::OpenMP::Task generate_join_task(const Nodecl::NodeclBase& enclosing_stmt);
};
class TransformNonVoidFunctionCalls : public Nodecl::ExhaustiveVisitor<void>
{
private:
bool _task_expr_optim_disabled;
bool _ignore_template_functions;
int _optimized_task_expr_counter;
int _new_return_vars_counter;
Nodecl::NodeclBase _enclosing_stmt;
std::shared_ptr<FunctionTaskSet> _function_task_set;
std::map<TL::Symbol, TL::Symbol> _transformed_task_map;
TL::ObjectList<Nodecl::FunctionCall> _ignore_these_function_calls;
TL::ObjectList<Nodecl::NodeclBase> _enclosing_stmts_with_more_than_one_task;
std::map<Nodecl::NodeclBase, Nodecl::NodeclBase> _funct_call_to_enclosing_stmt_map;
std::map<Nodecl::NodeclBase, Nodecl::NodeclBase> _enclosing_stmt_to_original_stmt;
std::map<Nodecl::NodeclBase, std::set<TL::Symbol> > _enclosing_stmt_to_return_vars_map;
public:
TransformNonVoidFunctionCalls(std::shared_ptr<FunctionTaskSet> function_task_set,
bool task_expr_optim_disabled,
bool ignore_template_functions);
virtual void visit(const Nodecl::FunctionCode& node);
virtual void visit(const Nodecl::TemplateFunctionCode& node);
virtual void visit(const Nodecl::ObjectInit& object_init);
virtual void visit(const Nodecl::ReturnStatement& return_stmt);
virtual void visit(const Nodecl::FunctionCall& func_call);
std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>& get_function_call_to_enclosing_stmt_map();
std::map<Nodecl::NodeclBase, Nodecl::NodeclBase>& get_enclosing_stmt_to_original_stmt_map();
std::map<Nodecl::NodeclBase, std::set<TL::Symbol> >& get_enclosing_stmt_to_return_variables_map();
void remove_nonvoid_function_tasks_from_function_task_set();
private:
Nodecl::NodeclBase get_enclosing_stmt(Nodecl::NodeclBase function_call);
bool only_one_task_is_involved_in_this_stmt(Nodecl::NodeclBase stmt);
void transform_object_init(Nodecl::NodeclBase enclosing_stmt, const locus_t* locus);
void transform_return_statement(Nodecl::NodeclBase enclosing_stmt, const locus_t* locus);
void transform_task_expression_into_simple_task(
Nodecl::FunctionCall func_call,
Nodecl::NodeclBase enclosing_stmt);
void add_new_task_to_the_function_task_set(
TL::Symbol ori_funct,
TL::Symbol new_funct,
const FunctionTaskInfo& ori_funct_task_info,
Nodecl::NodeclBase func_call,
bool has_return_argument,
TL::OpenMP::DependencyDirection dep_dir);
Nodecl::NodeclBase create_body_for_new_function_task(
TL::Symbol original_function,
TL::Symbol new_function);
Nodecl::NodeclBase create_body_for_new_optmized_function_task(
TL::Symbol ori_funct,
TL::Symbol new_funct,
Nodecl::NodeclBase enclosing_stmt,
Nodecl::NodeclBase new_funct_body,
TL::ObjectList<TL::Symbol>& captured_value_symbols);
TL::ObjectList<TL::Symbol> generate_list_of_symbols_to_be_captured(
Nodecl::NodeclBase rhs_expr,
Nodecl::NodeclBase lhs_expr,
Nodecl::NodeclBase func_call);
void update_rhs_expression(
Nodecl::NodeclBase node,
Nodecl::NodeclBase lhs_expr,
TL::Symbol ori_funct,
const ObjectList<TL::Symbol>& new_funct_related_symbols,
Nodecl::Utils::SimpleSymbolMap& map_for_captures,
Nodecl::NodeclBase& task_call);
};
}}
#endif 
