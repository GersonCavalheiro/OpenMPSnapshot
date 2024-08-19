#include "hlt-task-aggregation.hpp"
#include "hlt-task-aggregation-common.hpp"
#include "tl-omp.hpp"
#include "hlt-unroll-omp.hpp"
using namespace TL;
using namespace HLT;
using namespace OpenMP;
TaskAggregation::TaskAggregation(Statement stmt, AggregationMethod method)
: _stmt(stmt),
_method(method),
_bundling_amount(4),
_do_not_create_tasks(false),
_timing(false),
_global_bundling_src(NULL),
_finish_bundling_src(NULL),
_enclosing_function_def_tree(NULL)
{
}
TaskAggregation& TaskAggregation::set_aggregation_method(AggregationMethod method)
{
_method = method;
return *this;
}
TaskAggregation& TaskAggregation::set_bundling_amount(int amount)
{
_bundling_amount = amount;
return *this;
}
Source TaskAggregation::get_source()
{
bool contains_conditional_code = false;
if (!contains_relevant_openmp(_stmt))
{
return _stmt.prettyprint();
}
else
{
return do_aggregation(contains_conditional_code);
}
}
Source TaskAggregation::do_aggregation(bool contains_conditional_code)
{
{
switch ((int)_method)
{
case PREDICATION:
{
return do_predicated_aggregation();
}
case BUNDLING:
{
return do_bundled_aggregation();
}
default:
return Source("");
}
}
}
bool TaskAggregation::contains_relevant_openmp(Statement stmt)
{
bool b = false;
return TaskAggregation::contains_relevant_openmp(stmt, b);
}
bool TaskAggregation::contains_relevant_openmp(Statement stmt, bool &contains_conditional_code)
{
if (is_pragma_custom_construct("omp", "task", stmt.get_ast(), stmt.get_scope_link()))
return true;
else if (stmt.is_compound_statement())
{
ObjectList<Statement> stmt_list = stmt.get_inner_statements();
for (ObjectList<Statement>::iterator it = stmt_list.begin();
it != stmt_list.end();
it++)
{
if (contains_relevant_openmp(*it))
return true;
}
}
else if (IfStatement::predicate(stmt.get_ast()))
{
contains_conditional_code = true;
IfStatement if_statement(stmt.get_ast(), stmt.get_scope_link());
return contains_relevant_openmp(if_statement.get_then_body())
|| (if_statement.has_else() && contains_relevant_openmp(if_statement.get_else_body()));
}
return false;
}
void TaskAggregation::get_task_parts_aux(ObjectList<TaskPart>& result, ObjectList<Statement> &current_prologue, Statement stmt)
{
if (is_pragma_custom_construct("omp", "task", stmt.get_ast(), stmt.get_scope_link()))
{
PragmaCustomConstruct task_construct(stmt.get_ast(), stmt.get_scope_link());
TaskPart new_task_part(current_prologue, task_construct);
result.append(new_task_part);
current_prologue.clear();
}
else if (stmt.is_compound_statement())
{
ObjectList<Statement> stmt_list = stmt.get_inner_statements();
for (ObjectList<Statement>::iterator it = stmt_list.begin();
it != stmt_list.end();
it++)
{
get_task_parts_aux(result, current_prologue, *it);
}
}
else
{
current_prologue.append(stmt);
}
}
ObjectList<TaskPart> TaskAggregation::get_task_parts(Statement stmt)
{
ObjectList<TaskPart> result;
ObjectList<Statement> prologue;
get_task_parts_aux(result, prologue, stmt);
if (!prologue.empty())
{
TaskPart last_part(prologue);
result.append(last_part);
}
return result;
}
TaskAggregation& TaskAggregation::set_global_bundling_source(Source& src)
{
_global_bundling_src = &src;
return *this;
}
TaskAggregation& TaskAggregation::set_finish_bundling_source(Source& src)
{
_finish_bundling_src = &src;
return *this;
}
TaskAggregation& TaskAggregation::set_enclosing_function_tree(AST_t ast)
{
_enclosing_function_def_tree = ast;
return *this;
}
TaskAggregation& TaskAggregation::set_do_not_create_tasks(bool b)
{
_do_not_create_tasks = b;
return *this;
}
TaskAggregation& TaskAggregation::set_timing(bool b)
{
_timing = b;
return *this;
}
