#include "cxx-diagnostic.h"
#include "tl-analysis-utils.hpp"
#include "tl-datareference.hpp"
#include "tl-omp-lint.hpp"
#include "tl-tribool.hpp"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
namespace TL {
namespace OpenMP {
namespace {
static bool ompss_mode_enabled = false;
std::string log_file_path;
char log_file_name[PATH_MAX+1];
std::string usr_name;
static FILE* log_file = NULL;
#define CORRECTNESS_WARN_TYPE_LIST \
CORRECTNESS_WARN_TYPE(FP_Dead) \
CORRECTNESS_WARN_TYPE(P_Dead) \
CORRECTNESS_WARN_TYPE(FP_Incoherent) \
CORRECTNESS_WARN_TYPE(P_Incoherent) \
CORRECTNESS_WARN_TYPE(IN_Incoherent) \
CORRECTNESS_WARN_TYPE(IN_Pointed_Incoherent) \
CORRECTNESS_WARN_TYPE(OUT_Incoherent) \
CORRECTNESS_WARN_TYPE(OUT_Pointed_Incoherent) \
CORRECTNESS_WARN_TYPE(Race) \
CORRECTNESS_WARN_TYPE(SharedAutoStorage) \
CORRECTNESS_WARN_TYPE(Unused)
enum CorrectnessWarn_type {
#undef CORRECTNESS_WARN_TYPE
#define CORRECTNESS_WARN_TYPE(X) __##X,
CORRECTNESS_WARN_TYPE_LIST
#undef CORRECTNESS_WARN_TYPE
};
inline std::string correctness_warn_type_str(CorrectnessWarn_type cwt)
{
std::string warn_t = "";
switch(cwt)
{
#undef CORRECTNESS_WARN_TYPE
#define CORRECTNESS_WARN_TYPE(X) case __##X : return #X;
CORRECTNESS_WARN_TYPE_LIST
#undef CORRECTNESS_WARN_TYPE
default: WARNING_MESSAGE( "Unexpected type of correctness warning type '%d'", cwt );
};
return warn_t;
}
std::string get_nodecl_list_str( const Nodecl::List& nodecl_list )
{
std::string result;
for( Nodecl::List::const_iterator it = nodecl_list.begin( ); it != nodecl_list.end( ); )
{
result += it->prettyprint( );
++it;
if( it != nodecl_list.end( ) )
result += ", ";
}
return result;
}
std::string get_nodecl_map_str(const TL::Analysis::NodeclTriboolMap& nodecl_map)
{
std::string result;
for (TL::Analysis::NodeclTriboolMap::const_iterator it = nodecl_map.begin();
it != nodecl_map.end();)
{
if (it->second.is_true())
result += it->first.prettyprint();
++it;
if (it != nodecl_map.end())
result += ", ";
}
return result;
}
inline void print_warn_to_file(const Nodecl::NodeclBase& task, CorrectnessWarn_type warn_t, const std::string& vars)
{
if(!log_file_path.empty())
{
std::stringstream line_ss; line_ss << task.get_line();
std::string log = usr_name + " # " 
+ task.get_filename() + " # "
+ line_ss.str() + " # " 
+ correctness_warn_type_str(warn_t) 
+ vars + "\n";
if(fputs(log.c_str(), log_file) == EOF)
internal_error("Unable to write to file '%s', to store a correctness log.", log_file_name);
}
}
void get_message_common_info(
TL::Analysis::Node* task, 
std::string& task_label, 
std::string& tabulation)
{
TL::Analysis::PCFGPragmaInfo pragma_info(task->get_pragma_node_info());
if(pragma_info.has_clause(NODECL_OMP_SS_TASK_LABEL))
task_label = "::" + pragma_info.get_clause(NODECL_OMP_SS_TASK_LABEL).as<Nodecl::OmpSs::TaskLabel>().get_text();
tabulation = std::string(task_label.size(), ' ');
}
std::string get_auto_storage_message(bool use_plural, TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(use_plural)
{
return task_label + "local variables '%s' are shared, "
"but their lifetime may have ended when the task is executed.\n" +
tabulation + "Consider privatizing them or synchronizing the task before the local data is deallocated.\n";
}
else
{
return task_label + "local variable '%s' is shared, "
"but its lifetime may have ended when the task is executed.\n" +
tabulation + "Consider privatizing the variable or synchronizing the task before the local data is deallocated.\n";
}
}
std::string get_race_message(bool use_plural, TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(use_plural)
{
return task_label + "variables '%s' are in a race condition due to a concurrent usage.\n" + 
tabulation + "Consider synchronizing all concurrent accesses or privatizing the variables.\n";
}
else
{
return task_label + "variable '%s' is in a race condition due to a concurrent usage.\n" + 
tabulation + "Consider synchronizing all concurrent accesses or privatizing the variable.\n";
}
}
std::string get_dead_vars_message(bool use_plural, std::string data_sharing_atr, TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(use_plural)
{
return task_label + "variables '%s' are " + data_sharing_atr + ", " +
"therefore, updates on these variables will not be visible after the task.\n" +
tabulation + "Consider defining them as shared.\n";
}
else
{
return task_label + "variable '%s' is " + data_sharing_atr + ", " +
"therefore, updates on this variable will not be visible after the task.\n" +
tabulation + "Consider defining it as shared.\n";
}
}
std::string get_unused_scope_message(bool use_plural, TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(use_plural)
{
return task_label + "variables '%s' are not used within the task but they have been scoped.\n" +
tabulation + "This may slow down your application. Consider removing the data-sharing attributes.\n";
}
else
{
return task_label + "variable '%s' is not used within the task but it has been scoped.\n" +
tabulation + "This may slow down your application. Consider removing the data-sharing attribute.\n";
}
}
std::string get_incoherent_private_message(bool use_plural, TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(use_plural)
{
return task_label + "variables '%s' are private in the task, " +
"but their input value would have been used in a serial execution.\n" +
tabulation + "Consider defining them as firstprivate instead, to capture the initial value.\n";
}
else
{
return task_label + "variable '%s' is private in the task, " +
"but its input value would have been used in a serial execution.\n" +
tabulation + "Consider defining it as firstprivate instead, to capture the initial value.\n";
}
}
std::string get_incoherent_firstprivate_message(bool use_plural, TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(use_plural)
{
return task_label + "variables '%s' are firstprivate in the task, " +
"but their input value is never read.\n" +
tabulation + "Consider defining them as private instead.\n";
}
else
{
return task_label + "variable '%s' is firstprivate in the task, " +
"but its input value is never read.\n" +
tabulation + "Consider defining it as private instead\n";
}
}
std::string get_incoherent_in_deps_message(
bool use_plural,
bool pointed_obj_used, 
TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(pointed_obj_used)
{
if(use_plural)
{
return task_label + "variables '%s' are IN dependences, " +
"but their input values are not read in the task.\n" +
tabulation + "Instead, the object pointed by these variables is read.\n" +
tabulation + "Consider defining the dependencies on the pointed objects.\n";
}
else
{
return task_label + "variable '%s' is an IN dependency, " +
"but its input value is not read in the task.\n" +
tabulation + "Instead, the object pointed by this variable is read\n" +
tabulation + "Consider defining the dependency on the pointed object.\n";
}
}
else
{
if(use_plural)
{
return task_label + "variables '%s' are IN dependences, " +
"but their input values are not read in the task.\n" +
tabulation + "Consider removing these dependencies.\n";
}
else
{
return task_label + "variable '%s' is an IN dependency, " +
"but its input value is not read in the task.\n" +
tabulation + "Consider removing this dependency.\n";
}
}
}
std::string get_incoherent_out_deps_message(
bool use_plural,
bool pointed_obj_used, 
TL::Analysis::Node* task)
{
std::string task_label, tabulation;
get_message_common_info(task, task_label, tabulation);
if(pointed_obj_used)
{
if(use_plural)
{
return task_label + "variables '%s' are OUT dependences, " +
"but they are not written in the task.\n" +
tabulation + "Instead, the object pointed by these variables is written.\n" +
tabulation + "Consider defining the dependencies on the pointed objects.\n";
}
else
{
return task_label + "variable '%s' is an OUT dependency, " +
"but it is not written in the task.\n" +
tabulation + "Instead, the object pointed by this variable is written\n" +
tabulation + "Consider defining the dependency on the pointed object.\n";
}
}
else
{
if(use_plural)
{
return task_label + "variables '%s' are OUT dependences, " +
"but they are not written in the task.\n" +
tabulation + "Consider removing this dependency.\n";
}
else
{
return task_label + "variable '%s' is an OUT dependency, " +
"but it is not written in the task.\n" +
tabulation + "Consider removing this dependency.\n";
}
}
}
tribool symbol_is_local(const Nodecl::NodeclBase& nodecl_sym, Nodecl::List& local_syms)
{
TL::Symbol sym = nodecl_sym.get_symbol();
if (!sym.is_valid())
{
return tribool();
}
tribool result = !sym.get_type().is_any_reference() &&
sym.get_scope().is_block_scope();
if( result.is_true( ) )
local_syms.append( nodecl_sym );
return result;
}
tribool any_symbol_is_local(const TL::Analysis::NodeclSet& item_list, Nodecl::List& local_syms)
{
tribool result( false );
for (TL::Analysis::NodeclSet::const_iterator it = item_list.begin(); it != item_list.end(); it++)
result = result || symbol_is_local(*it, local_syms);
return result;
}
tribool task_is_locally_bound( TL::Analysis::Node *n, Nodecl::List& local_vars )
{
ERROR_CONDITION( !n->is_omp_task_node( ), "Expecting a Task node, but found a '%s' node.", 
n->get_type_as_string( ).c_str( ) );
Nodecl::NodeclBase task = n->get_graph_related_ast( );
ERROR_CONDITION( task.is_null( ), "Invalid target task tree related to node %d.", n->get_id( ) );
const TL::Analysis::NodeclSet& shared_vars = n->get_all_shared_accesses();
return any_symbol_is_local(shared_vars, local_vars);
}
tribool task_only_synchronizes_in_enclosing_scopes(
TL::Analysis::Node *task, 
const Nodecl::List& local_vars)
{
TL::Analysis::Node* task_creation = TL::Analysis::ExtensibleGraph::get_task_creation_from_task(task);
ERROR_CONDITION (task_creation==NULL, 
"Task creation of node %d not found in its list of parents.\n", 
task->get_id());
const TL::ObjectList<TL::Analysis::Node*>& children = task->get_children();
for(Nodecl::List::const_iterator itv = local_vars.begin(); itv != local_vars.end(); ++itv)
{
Scope var_sc(TL::Analysis::Utils::get_nodecl_base(*itv).get_symbol().get_scope());
TL::Analysis::Node* var_ctx_node = NULL;
TL::Analysis::Node* task_outer = task_creation->get_outer_node();
while(task_outer != NULL)
{
if(task_outer->is_context_node())
{
const Nodecl::NodeclBase& ctx = task_outer->get_graph_related_ast();
Scope ctx_sc(ctx.retrieve_context());
if((ctx_sc == var_sc) || var_sc.scope_is_enclosed_by(ctx_sc))
{   
var_ctx_node = task_outer;
goto check_sync;
}
}
task_outer = task_outer->get_outer_node();
}
return false;
check_sync:
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator it = children.begin();
it != children.end(); ++it)
{
if((*it)->is_omp_virtual_tasksync())
return false;
TL::Analysis::Node* sync = ((*it)->is_omp_task_node() ? TL::Analysis::ExtensibleGraph::get_task_creation_from_task(*it) 
: *it);
if(!TL::Analysis::ExtensibleGraph::node_contains_node(var_ctx_node, sync))
return false;
}
}
return true;
}
typedef std::map<Nodecl::NodeclBase, TL::ObjectList<TL::Analysis::Node*>, Nodecl::Utils::Nodecl_structural_less> VarToNodesMap;
void compute_usage_between_nodes(
TL::Analysis::Node* current,
TL::Analysis::Node* ini,
TL::Analysis::Node* fin,
TL::Analysis::Node* omitted_node,
bool skip_other_tasks,
const TL::Analysis::NodeclSet& all_data_refs,
VarToNodesMap& used_data_refs_to_node)
{
if (current->is_visited() || current == fin)
return;
current->set_visited(true);
if (current != omitted_node)
{
if (current->is_graph_node())
{   
compute_usage_between_nodes(
current->get_graph_entry_node(),
ini,
fin,
omitted_node,
skip_other_tasks,
all_data_refs,
used_data_refs_to_node);
}
else if (current->has_statements())
{
const TL::Analysis::NodeclSet& ue = current->get_ue_vars();
const TL::Analysis::NodeclSet& kill = current->get_killed_vars();
const TL::Analysis::NodeclSet& undef = current->get_undefined_behaviour_vars();
TL::Analysis::NodeclSet accessed_vars;
accessed_vars.insert(ue.begin(), ue.end());
accessed_vars.insert(kill.begin(), kill.end());
accessed_vars.insert(undef.begin(), undef.end());
for (TL::Analysis::NodeclSet::const_iterator it = all_data_refs.begin();
it != all_data_refs.end(); ++it)
{
if (TL::Analysis::Utils::nodecl_set_contains_nodecl(*it, accessed_vars)
|| !TL::Analysis::Utils::nodecl_set_contains_enclosing_nodecl(*it, accessed_vars).is_null()
|| !TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(*it, accessed_vars).is_null())
{
if (used_data_refs_to_node.find(*it) != used_data_refs_to_node.end())
used_data_refs_to_node[*it].insert(current);
else
used_data_refs_to_node[*it] = TL::ObjectList<TL::Analysis::Node*>(1, current);
}
}
}
}
TL::ObjectList<TL::Analysis::Node*> children;
if (current->is_exit_node())
{   
TL::Analysis::Node* outer = current->get_outer_node();
if (outer != NULL )
children = current->get_outer_node()->get_children();
}
else
{
children = current->get_children();
}
for (TL::ObjectList<TL::Analysis::Node*>::iterator it = children.begin(); it != children.end(); ++it)
{
if (!skip_other_tasks || !(*it)->is_omp_task_node())
{   
compute_usage_between_nodes(
*it,
ini,
fin,
omitted_node,
skip_other_tasks,
all_data_refs,
used_data_refs_to_node);
}
}
}
bool check_concurrency(
TL::Analysis::Node* task,
const Nodecl::NodeclBase& n,
const TL::ObjectList<TL::Analysis::Node*>& task_nodes,
const TL::ObjectList<TL::Analysis::Node*>& concurrent_nodes,
const TL::Analysis::NodeclSet& task_defs,
tribool certainty,
TL::Analysis::NodeclSet& warned_vars,
TL::Analysis::NodeclTriboolMap& race_vars_certainty)
{
bool race = false;
TL::Analysis::NodeclSet concurrent_defs;
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator it = concurrent_nodes.begin();
it != concurrent_nodes.end(); ++it)
{
const TL::Analysis::NodeclSet& node_killed = (*it)->get_killed_vars();
concurrent_defs.insert(node_killed.begin(), node_killed.end());
}
if ((!task_defs.empty()
&& (TL::Analysis::Utils::nodecl_set_contains_nodecl(n, task_defs)
|| !TL::Analysis::Utils::nodecl_set_contains_enclosing_nodecl(n, task_defs).is_null()
|| !TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(n, task_defs).is_null()))
|| (!concurrent_defs.empty()
&& (TL::Analysis::Utils::nodecl_set_contains_nodecl(n, concurrent_defs)
|| !TL::Analysis::Utils::nodecl_set_contains_enclosing_nodecl(n, concurrent_defs).is_null()
|| !TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(n, concurrent_defs).is_null())))
{
TL::ObjectList<TL::Analysis::Node*> all_nodes = task_nodes;
all_nodes.insert(concurrent_nodes);
for (TL::ObjectList<TL::Analysis::Node*>::iterator it = all_nodes.begin();
it != all_nodes.end(); ++it)
{
if (!TL::Analysis::ExtensibleGraph::node_is_in_synchronous_construct(*it))
{
race = true;
race_vars_certainty[n] = certainty;
task->add_correctness_race_var(n, certainty);
warned_vars.insert(n);
break;
}
}
}
return race;
}
static bool is_global_var(
const Nodecl::NodeclBase& v,
TL::Analysis::ExtensibleGraph* pcfg)
{
const TL::Analysis::NodeclSet& global_vars = pcfg->get_global_variables();
Nodecl::NodeclBase v_base = TL::Analysis::Utils::get_nodecl_base(v);
return (global_vars.find(v) != global_vars.end()
|| (!v_base.is_null()
&& global_vars.find(v_base) != global_vars.end()));
}
static void get_next_sync_for_sequential_code(
ObjectList<TL::Analysis::Node*>& seq_next_sync,
TL::Analysis::ExtensibleGraph* pcfg,
TL::Analysis::Node* task)
{
const TL::ObjectList<TL::Analysis::Node*>& next_sync = pcfg->get_task_next_sync_for_sequential_code(task);
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator itn = next_sync.begin();
itn != next_sync.end(); ++itn)
{
TL::ObjectList<TL::Analysis::Node*> children(1, *itn);
std::set<TL::Analysis::Node*> treated_children;
while (!children.empty())
{
TL::Analysis::Node* n = children.back();
children.pop_back();
if (treated_children.find(n) != treated_children.end())
continue;
treated_children.insert(n);
if (n->is_omp_taskwait_node() || n->is_omp_barrier_graph_node())
{
if (!seq_next_sync.contains(n))
seq_next_sync.insert(n);
}
else if (n->is_omp_virtual_tasksync())
{
TL::Analysis::Node* n_sync = pcfg->get_graph()->get_graph_exit_node();
if (!seq_next_sync.contains(n_sync))
seq_next_sync.insert(n_sync);
}
else if(n->is_omp_task_node())
{
const TL::ObjectList<TL::Analysis::Node*>& n_children = n->get_children();
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator itc = n_children.begin();
itc != n_children.end(); ++itc)
{
children.prepend(*itc);
}
}
else
{
if (!seq_next_sync.contains(n))
seq_next_sync.insert(n);
}
}
}
}
static TL::Analysis::Node* get_variable_context(
const Nodecl::NodeclBase& data_ref,
TL::Analysis::Node* n,
TL::Analysis::ExtensibleGraph* pcfg)
{
TL::Scope data_ref_sc(data_ref.retrieve_context());
TL::Analysis::Node* data_ref_ctx = NULL;
TL::Analysis::Node* outer = n->get_outer_node();
while (data_ref_ctx == NULL && outer != NULL)
{
if (outer->is_context_node()
&& data_ref_sc == outer->get_graph_related_ast().retrieve_context())
{
data_ref_ctx = outer;
}
outer = (outer->is_omp_task_node()
? TL::Analysis::ExtensibleGraph::get_task_creation_from_task(outer)->get_outer_node()
: outer->get_outer_node());
}
if (data_ref_ctx == NULL)
data_ref_ctx = pcfg->get_graph();
return data_ref_ctx;
}
static TL::Analysis::Node* get_enclosing_private_concurrent_context(
const Nodecl::NodeclBase& var,
TL::Analysis::Node* n,
TL::Analysis::Node* most_outer_sc,
TL::Analysis::ExtensibleGraph* pcfg)
{
TL::Analysis::Node* private_sc = NULL;
TL::Analysis::Node* outer = n->get_outer_node();
while (private_sc == NULL && outer != most_outer_sc)
{
if (outer->is_omp_parallel_node() || outer->is_omp_task_node())
{
const TL::Analysis::NodeclSet& outer_private_vars = outer->get_all_private_vars();
if (outer_private_vars.find(var) != outer_private_vars.end())
{
private_sc = outer;
}
}
outer = (outer->is_omp_task_node()
? TL::Analysis::ExtensibleGraph::get_task_creation_from_task(outer)->get_outer_node()
: outer->get_outer_node());
}
return private_sc;
}
static TL::Analysis::Node* get_enclosing_parallel_context(
TL::Analysis::Node* n,
TL::Analysis::Node* most_outer_sc,
TL::Analysis::ExtensibleGraph* pcfg)
{
TL::Analysis::Node* parallel_sc = NULL;
TL::Analysis::Node* outer = n->get_outer_node();
while (outer != most_outer_sc)
{
if (outer->is_omp_parallel_node())
{
parallel_sc = outer;
}
outer = (outer->is_omp_task_node()
? TL::Analysis::ExtensibleGraph::get_task_creation_from_task(outer)->get_outer_node()
: outer->get_outer_node());
}
return parallel_sc;
}
static TL::Analysis::Node* get_enclosing_loop_context(
TL::Analysis::Node* n,
TL::Analysis::Node* most_outer_sc,
TL::Analysis::ExtensibleGraph* pcfg)
{
TL::Analysis::Node* loop_sc = NULL;
TL::Analysis::Node* outer = n->get_outer_node();
while (outer != most_outer_sc)
{
if (outer->is_loop_node())
loop_sc = outer;
outer = (outer->is_omp_task_node()
? TL::Analysis::ExtensibleGraph::get_task_creation_from_task(outer)->get_outer_node()
: outer->get_outer_node());
}
return loop_sc;
}
static void get_shared_vars_between_limits(
TL::Analysis::Node* task,
TL::Analysis::Node* scope,
const TL::ObjectList<TL::Analysis::Node*>& last_syncs,
const TL::ObjectList<TL::Analysis::Node*>& next_syncs,
const TL::Analysis::NodeclSet& task_shared_variables,
std::map<TL::Analysis::Node*, VarToNodesMap>& treated_scopes,
VarToNodesMap& result)
{
if (scope != NULL && treated_scopes.find(scope) != treated_scopes.end())
{
result = treated_scopes[scope];
}
else
{
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator itl = last_syncs.begin();
itl != last_syncs.end(); ++itl)
{
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator itn = next_syncs.begin();
itn != next_syncs.end(); ++itn)
{
compute_usage_between_nodes(
*itl,
*itl,
*itn,
task,
true,
task_shared_variables,
result);
}
}
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator itl = last_syncs.begin();
itl != last_syncs.end(); ++itl)
{
TL::Analysis::ExtensibleGraph::clear_visits(*itl);
}
if (scope != NULL)
treated_scopes[scope] = result;
}
}
bool var_is_defined_between_nodes(
TL::Analysis::Node* source,
TL::Analysis::Node* target,
const Nodecl::NodeclBase& n,
bool& is_defined)
{
if (source == target)
return true;
if(source->is_visited_aux())
return false;
source->set_visited_aux(true);
const TL::Analysis::NodeclSet& killed_vars = source->get_killed_vars();
if (TL::Analysis::Utils::nodecl_set_contains_nodecl(n, killed_vars))
is_defined = true;
bool target_found = false;
TL::ObjectList<TL::Analysis::Node*> children = source->get_children();
if (source->is_exit_node())
{
TL::Analysis::Node* outer = source->get_outer_node();
if (outer != NULL)
children = outer->get_children();
}
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator it = children.begin();
it != children.end() && !target_found; ++it)
{
TL::Analysis::Node* c = *it;
if (!c->is_omp_task_node())
{
if (c->is_graph_node())
{
c->set_visited_aux(true);
c = c->get_graph_entry_node();
}
target_found = var_is_defined_between_nodes(c, target, n, is_defined);
}
}
return target_found;
}
tribool task_may_cause_race_condition(
TL::Analysis::ExtensibleGraph* pcfg,
TL::Analysis::Node* task,
std::map<TL::Analysis::Node*, VarToNodesMap>& treated_scopes,
TL::Analysis::NodeclTriboolMap& race_vars_certainty)
{
tribool result = tribool::False;
const TL::Analysis::NodeclSet& task_shared_variables = task->get_all_shared_accesses();
ObjectList<TL::Analysis::Node*> seq_next_sync;
get_next_sync_for_sequential_code(seq_next_sync, pcfg, task);
VarToNodesMap task_vars_to_nodes;
TL::Analysis::Node* task_entry = task->get_graph_entry_node();
compute_usage_between_nodes(
task_entry,
task_entry,
task->get_graph_exit_node(),
NULL,
true,
task_shared_variables,
task_vars_to_nodes);
TL::Analysis::ExtensibleGraph::clear_visits_in_level(task_entry, task);
TL::Analysis::NodeclSet& task_defs = task->get_killed_vars();
TL::Analysis::Node* task_creation = TL::Analysis::ExtensibleGraph::get_task_creation_from_task(task);
TL::Analysis::NodeclSet warned_vars;
for (TL::Analysis::NodeclSet::const_iterator it = task_shared_variables.begin();
it != task_shared_variables.end(); ++it)
{
const Nodecl::NodeclBase& n = *it;
Nodecl::NodeclBase n_base = TL::Analysis::Utils::get_nodecl_base(n);
Type t(it->get_type());
if ((task_vars_to_nodes.find(n) == task_vars_to_nodes.end())
|| t.is_const())
continue;
const TL::ObjectList<TL::Analysis::Node*>& task_nodes = task_vars_to_nodes[n];
TL::Analysis::Node* n_sc = get_variable_context(n_base, task_creation, pcfg);
TL::Analysis::Node* private_sc = get_enclosing_private_concurrent_context(n_base, task_creation, n_sc, pcfg);
TL::Analysis::Node* loop_sc = get_enclosing_loop_context(task_creation, private_sc, pcfg);
VarToNodesMap in_loop_vars_to_nodes;
if (loop_sc != NULL)
{
get_shared_vars_between_limits(
task,
loop_sc,
ObjectList<TL::Analysis::Node*>(1, loop_sc->get_graph_entry_node()),
seq_next_sync,
task_shared_variables,
treated_scopes,
in_loop_vars_to_nodes);
}
TL::Analysis::Node* parallel_sc = get_enclosing_parallel_context(task_creation, private_sc, pcfg);
VarToNodesMap in_parallel_vars_to_nodes;
if (parallel_sc != NULL)
{
get_shared_vars_between_limits(
task,
parallel_sc,
ObjectList<TL::Analysis::Node*>(1, parallel_sc->get_graph_entry_node()),
ObjectList<TL::Analysis::Node*>(1, parallel_sc->get_graph_exit_node()),
task_shared_variables,
treated_scopes,
in_parallel_vars_to_nodes);
}
VarToNodesMap in_function_vars_to_nodes;
if (parallel_sc==NULL && loop_sc==NULL)
{
get_shared_vars_between_limits(
task,
NULL,
ObjectList<TL::Analysis::Node*>(1, task_creation),
seq_next_sync,
task_shared_variables,
treated_scopes,
in_function_vars_to_nodes);
}
if (private_sc != NULL)
{   
VarToNodesMap concurrent_vars_to_nodes;
if (treated_scopes.find(private_sc) == treated_scopes.end())
{   
if (private_sc->is_omp_task_node())
{
TL::Analysis::Node* last_sync = (loop_sc==NULL ? task_creation : loop_sc->get_graph_entry_node());
for (TL::ObjectList<TL::Analysis::Node*>::iterator itsn = seq_next_sync.begin();
itsn != seq_next_sync.end(); ++itsn)
{
compute_usage_between_nodes(
last_sync,
last_sync,
*itsn,
task,
false,
task_shared_variables,
concurrent_vars_to_nodes);
}
}
else
{
TL::Analysis::Node* last_sync = private_sc->get_graph_entry_node();
TL::Analysis::Node* next_sync = private_sc->get_graph_exit_node();
compute_usage_between_nodes(
last_sync,
last_sync,
next_sync,
task,
false,
task_shared_variables,
concurrent_vars_to_nodes);
}
}
else
{
concurrent_vars_to_nodes = treated_scopes[private_sc];
}
if (concurrent_vars_to_nodes.find(n) != concurrent_vars_to_nodes.end())
{
const TL::ObjectList<TL::Analysis::Node*>& concurrent_nodes = concurrent_vars_to_nodes[n];
result = result || check_concurrency(
task,
n,
task_nodes,
concurrent_nodes,
task_defs,
tribool::True,
warned_vars,
race_vars_certainty);
}
}
else
{   
const TL::ObjectList<TL::Analysis::Node*>& in_parallel_nodes = in_parallel_vars_to_nodes[n];
const TL::ObjectList<TL::Analysis::Node*>& in_loop_nodes = in_loop_vars_to_nodes[n];
bool tmp = false;
if (!in_parallel_nodes.empty())
{
tmp = check_concurrency(
task,
n,
task_nodes,
in_parallel_nodes,
task_defs,
tribool::True,
warned_vars,
race_vars_certainty);
}
if (!tmp && !in_loop_nodes.empty())
{
tmp = check_concurrency(
task,
n,
task_nodes,
in_loop_nodes,
task_defs,
tribool::True,
warned_vars,
race_vars_certainty);
}
if (!tmp
&& ((!n_base.is_null() && is_global_var(n_base, pcfg))      
|| (n_sc == NULL && t.is_any_reference())))     
{
race_vars_certainty[n] = tribool::Unknown;
task->add_correctness_race_var(n, tribool::Unknown);
warned_vars.insert(n);
}
result = result || tmp;
}
}
const TL::ObjectList<TL::Analysis::Node*>& concurrent_tasks = pcfg->get_task_concurrent_tasks(task);
for (ObjectList<TL::Analysis::Node*>::const_iterator it = concurrent_tasks.begin(); it != concurrent_tasks.end(); ++it)
{
TL::Analysis::Node* concurrent_task = *it;
VarToNodesMap in_concurrent_task_vars_to_nodes;
if (concurrent_task == task)
{
in_concurrent_task_vars_to_nodes = task_vars_to_nodes;
}
else
{
TL::Analysis::Node* concurrent_task_entry = concurrent_task->get_graph_entry_node();
compute_usage_between_nodes(
concurrent_task_entry,
concurrent_task_entry,
concurrent_task->get_graph_exit_node(),
task,
false,
task_shared_variables,
in_concurrent_task_vars_to_nodes);
TL::Analysis::ExtensibleGraph::clear_visits_in_level(concurrent_task_entry, concurrent_task);
}
for (TL::Analysis::NodeclSet::const_iterator itt = task_shared_variables.begin();
itt != task_shared_variables.end(); ++itt)
{
const Nodecl::NodeclBase& n = itt->no_conv();
if (n.is<Nodecl::ArraySubscript>())
{
TL::Analysis::Node* concurrent_task_creation = TL::Analysis::ExtensibleGraph::get_task_creation_from_task(concurrent_task);
TL::Analysis::Node* source = task_creation;
TL::Analysis::Node* target = concurrent_task_creation;
if (task_creation == concurrent_task_creation)
{
TL::ObjectList<TL::Analysis::Node*> conc_task_creation_children = concurrent_task_creation->get_children();
ERROR_CONDITION(conc_task_creation_children.size()!=2,
"Task creation %d has %d children, but 2 are expected.\n",
concurrent_task_creation->get_id(), conc_task_creation_children.size());
target = source;
source = conc_task_creation_children[0]->is_omp_task_node() ? conc_task_creation_children[1]
: conc_task_creation_children[0];
}
const TL::ObjectList<Nodecl::NodeclBase>& accessed_vars = Nodecl::Utils::get_all_memory_accesses(n);
bool def = false;
for (TL::ObjectList<Nodecl::NodeclBase>::const_iterator ittt = accessed_vars.begin();
ittt != accessed_vars.end(); ++ittt)
{
if (Nodecl::Utils::structurally_equal_nodecls(n, *ittt, true))
continue;
bool target_found = var_is_defined_between_nodes(source, target, *ittt, def);
TL::Analysis::ExtensibleGraph::clear_visits_aux(source);
if (target_found && def)
{
if (task_creation == concurrent_task_creation)
goto skip_current_var;
else
break;
}
}
if (task_creation != concurrent_task_creation
&& TL::Analysis::ExtensibleGraph::node_is_ancestor_of_node(target, source))
{
TL::Analysis::Node* tmp = source;
source = target;
target = tmp;
def = false;
for (TL::ObjectList<Nodecl::NodeclBase>::const_iterator ittt = accessed_vars.begin();
ittt != accessed_vars.end(); ++ittt)
{
if (Nodecl::Utils::structurally_equal_nodecls(n, *ittt, true))
continue;
bool target_found = var_is_defined_between_nodes(source, target, *ittt, def);
TL::Analysis::ExtensibleGraph::clear_visits_aux(source);
if (target_found && def)
goto skip_current_var;
}
}
}
{
const TL::ObjectList<TL::Analysis::Node*>& in_concurrent_task_nodes = in_concurrent_task_vars_to_nodes[n];
const TL::ObjectList<TL::Analysis::Node*>& task_nodes = task_vars_to_nodes[n];
result = result || check_concurrency(
task,
n,
task_nodes,
in_concurrent_task_nodes,
task_defs,
tribool::True,
warned_vars,
race_vars_certainty);
}
skip_current_var: ;
}
}
return result;
}
bool var_is_used_between_nodes(TL::Analysis::Node* source, TL::Analysis::Node* target, const Nodecl::NodeclBase& n)
{
if (source->is_visited() || (source == target) || source->is_exit_node())
return false;
source->set_visited(true);
if (source->is_graph_node())
{
if (var_is_used_between_nodes(source->get_graph_entry_node(), target, n))
return true;
}
else if(source->has_statements())
{
const TL::ObjectList<Nodecl::NodeclBase>& stmts = source->get_statements();
for (TL::ObjectList<Nodecl::NodeclBase>::const_iterator it = stmts.begin();
it != stmts.end(); ++it)
{
const TL::ObjectList<Nodecl::NodeclBase>& mem_accesses = Nodecl::Utils::get_all_memory_accesses(*it);
if (Nodecl::Utils::list_contains_nodecl_by_structure(mem_accesses, n))
{
return true;
}
}
}
const TL::ObjectList<TL::Analysis::Node*>& children = source->get_children();
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator it = children.begin();
it != children.end(); ++it)
{
if (var_is_used_between_nodes(*it, target, n))
return true;
}
return false;
}
#if 0
bool var_is_used_in_node_after_definition(TL::Analysis::Node* node, const Nodecl::NodeclBase& n)
{
bool result = false;
const TL::ObjectList<Nodecl::NodeclBase>& stmts = node->get_statements();
WritesVisitor wv;
TL::ObjectList<Nodecl::NodeclBase>::const_iterator it;
TL::ObjectList<Nodecl::NodeclBase>::const_iterator it2 = stmts.end();
for (it = stmts.begin(); it != stmts.end(); ++it)
{
wv.walk(*it);
const ObjectList<Nodecl::NodeclBase>& defined_syms = wv.get_defined_symbols();
if( Nodecl::Utils::list_contains_nodecl_by_structure(defined_syms, n))
it2 = it;
wv.clear();
}
if(it2 != stmts.end())
{
it = it2; it++;
for( ; it != stmts.end() && !result; ++it)
{
const TL::ObjectList<Nodecl::NodeclBase>& mem_accesses = Nodecl::Utils::get_all_memory_accesses( *it );
if (Nodecl::Utils::list_contains_nodecl_by_structure(mem_accesses, n))
result = true;
}
}
return result;
}
#endif
void get_var_last_definitions(
Nodecl::NodeclBase n, TL::Analysis::Node* task,
TL::ObjectList<Nodecl::NodeclBase>& last_definitions_vars,
TL::ObjectList<TL::Analysis::Node*>& last_definitions_nodes)
{
TL::ObjectList<TL::Analysis::Node*> result;
TL::Analysis::Node* task_entry = task->get_graph_entry_node();
TL::Analysis::Node* task_exit = task->get_graph_exit_node();
TL::ObjectList<TL::Analysis::Edge*> entries = task_exit->get_entry_edges();
std::set<TL::Analysis::Node*> visited_nodes;
while (!entries.empty())
{
TL::ObjectList<TL::Analysis::Edge*> new_entries;
for (TL::ObjectList<TL::Analysis::Edge*>::iterator it = entries.begin();
it != entries.end(); ++it)
{
TL::Analysis::Node* src = (*it)->get_source();
if (visited_nodes.find(src) != visited_nodes.end())
continue;
visited_nodes.insert(src);
if (src->is_graph_node())
src = src->get_graph_exit_node();
const Nodecl::List& killed_subparts =
TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(n, src->get_killed_vars());
if (!killed_subparts.is_null())
{
for (Nodecl::List::iterator itk = killed_subparts.begin(); itk != killed_subparts.end(); ++itk)
last_definitions_vars.append(*itk);
last_definitions_nodes.append(src);
}
else
{
if (src->is_entry_node() && src != task_entry)
new_entries.append(src->get_outer_node()->get_entry_edges());
else
new_entries.append(src->get_entry_edges());
}
}
entries = new_entries;
}
}
bool var_is_used_in_task_after_definition(const Nodecl::NodeclBase& n, TL::Analysis::Node* task)
{
TL::Analysis::Node* task_exit = task->get_graph_exit_node();
TL::ObjectList<Nodecl::NodeclBase> last_definitions_vars;
TL::ObjectList<TL::Analysis::Node*> last_definitions_nodes;
get_var_last_definitions(n, task, last_definitions_vars, last_definitions_nodes);
ERROR_CONDITION(last_definitions_vars.empty(),
"Variable '%s' is defined inside task %d, but the definition has not been found\n",
n.prettyprint().c_str(), task->get_id());
TL::ObjectList<Nodecl::NodeclBase>::const_iterator itv = last_definitions_vars.begin();
TL::ObjectList<TL::Analysis::Node*>::const_iterator itn = last_definitions_nodes.begin();
for (; itv != last_definitions_vars.end(); ++itv, ++itn)
{
const TL::ObjectList<TL::Analysis::Node*>& children = (*itn)->get_children();
for (TL::ObjectList<TL::Analysis::Node*>::const_iterator itc = children.begin();
itc != children.end(); ++itc)
{
TL::Analysis::Node* last_def = *itc;
bool result = var_is_used_between_nodes(last_def, task_exit, *itv);
TL::Analysis::ExtensibleGraph::clear_visits(last_def);
if (!result)
return false;
}
}
return true;
}
std::string get_dead_vars(
const Nodecl::List& var_list,
unsigned int& n_vars,
const TL::Analysis::NodeclSet& killed_vars,
TL::Analysis::Node* task)
{
std::string dead_code_vars;
for (Nodecl::List::iterator it = var_list.begin(); it != var_list.end(); ++it)
{
if (!TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(*it, killed_vars).is_null())
{
if (!var_is_used_in_task_after_definition(*it, task))
{
n_vars++;
task->add_correctness_dead_var(*it);
dead_code_vars += it->prettyprint() + ", ";
}
}
}
return dead_code_vars;
}
static bool is_used_address(const Nodecl::NodeclBase& n,
const TL::Analysis::NodeclSet& used_addrs)
{
for (TL::Analysis::NodeclSet::const_iterator it = used_addrs.begin();
it != used_addrs.end(); ++it)
{
const Nodecl::NodeclBase& base_it = TL::Analysis::Utils::get_nodecl_base(*it);
if (Nodecl::Utils::structurally_equal_nodecls(n, base_it, true))
return true;
}
return false;
}
std::string get_unnecessarily_scoped_vars(
const Nodecl::List& scope_vars, unsigned int& n_vars,
const TL::Analysis::NodeclSet& all_vars,
const TL::Analysis::NodeclSet& dependency_vars,
const TL::Analysis::NodeclSet& used_addrs,
TL::Analysis::Node* task)
{
std::string unnecessarily_scoped_vars;
const TL::Analysis::NodeclSet& used_addresses = task->get_used_addresses();
for (Nodecl::List::const_iterator it = scope_vars.begin(); it != scope_vars.end(); ++it)
{
if (TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(*it, all_vars).is_null()
&& (TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(*it, dependency_vars).is_null()
|| TL::Analysis::Utils::nodecl_set_contains_enclosing_nodecl(*it, dependency_vars).is_null())
&& TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(*it, used_addresses).is_null()
&& !is_used_address(*it, used_addrs))
{
std::string var_name = it->prettyprint();
if(it->get_symbol().is_saved_expression())
continue;
unnecessarily_scoped_vars += var_name + ", ";
task->add_correctness_unnecessarily_scoped_var(*it);
n_vars++;
}
}
return unnecessarily_scoped_vars;
}
std::string get_incoherent_private_vars(
const Nodecl::List& private_vars, unsigned int& n_vars,
const TL::Analysis::NodeclSet& all_ue_vars,
TL::Analysis::Node* task)
{
std::string incoherent_private_vars;
const Nodecl::List& correctness_dead_vars = task->get_correctness_dead_vars();
for (Nodecl::List::iterator it = private_vars.begin(); it != private_vars.end(); ++it)
{
const Nodecl::NodeclBase& n = *it;
if (TL::Analysis::Utils::nodecl_set_contains_nodecl(n, all_ue_vars)
&& !Nodecl::Utils::nodecl_is_in_nodecl_list(n, correctness_dead_vars))
{
incoherent_private_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_p_var(n);
n_vars++;
}
}
return incoherent_private_vars;
}
std::string get_incoherent_firstprivate_vars(
const Nodecl::List& firstprivate_vars, unsigned int& n_vars,
const TL::Analysis::NodeclSet& all_ue_vars,
const TL::Analysis::NodeclSet& dependency_vars,
const TL::Analysis::NodeclSet& used_addrs,
TL::Analysis::Node* task)
{
std::string incoherent_firstprivate_vars;
const TL::Analysis::NodeclSet& used_addresses = task->get_used_addresses();
const Nodecl::List& correctness_dead_vars = task->get_correctness_dead_vars();
for (Nodecl::List::iterator it = firstprivate_vars.begin(); it != firstprivate_vars.end(); ++it)
{
const Nodecl::NodeclBase& n = *it;
if (!TL::Analysis::Utils::nodecl_set_contains_nodecl(n, all_ue_vars)
&& TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(n, used_addresses).is_null()
&& !TL::Analysis::Utils::nodecl_set_contains_nodecl(n, dependency_vars)
&& !Nodecl::Utils::nodecl_is_in_nodecl_list(n, correctness_dead_vars)
&& !is_used_address(*it, used_addrs))
{
if(it->get_symbol().is_saved_expression())
continue;
incoherent_firstprivate_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_fp_var(n);
n_vars++;
}
}
return incoherent_firstprivate_vars;
}
void check_task_incoherent_data_sharing(TL::Analysis::Node* task)
{
Nodecl::List firstprivate_vars, private_vars, task_scoped_vars;
TL::Analysis::PCFGPragmaInfo task_pragma_info = task->get_pragma_node_info( );
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_FIRSTPRIVATE))
{
firstprivate_vars = task_pragma_info.get_clause(NODECL_OPEN_M_P_FIRSTPRIVATE).as<Nodecl::OpenMP::Firstprivate>().get_symbols().shallow_copy().as<Nodecl::List>();
task_scoped_vars.append( firstprivate_vars );
} 
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_PRIVATE))
{
private_vars = task_pragma_info.get_clause(NODECL_OPEN_M_P_PRIVATE).as<Nodecl::OpenMP::Private>().get_symbols().shallow_copy().as<Nodecl::List>();
task_scoped_vars.append( private_vars );
} 
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_SHARED))
{
Nodecl::List shared_vars = task_pragma_info.get_clause(NODECL_OPEN_M_P_SHARED).as<Nodecl::OpenMP::Shared>().get_symbols().shallow_copy().as<Nodecl::List>();
task_scoped_vars.append( shared_vars );
}
if (task_pragma_info.has_clause(NODECL_OMP_SS_SHARED_AND_ALLOCA))
{
Nodecl::List shared_alloca_vars = task_pragma_info.get_clause(NODECL_OMP_SS_SHARED_AND_ALLOCA).as<Nodecl::OmpSs::SharedAndAlloca>().get_exprs().shallow_copy().as<Nodecl::List>();
task_scoped_vars.append( shared_alloca_vars );
}
const TL::Analysis::NodeclSet& ue_vars = task->get_ue_vars();
const TL::Analysis::NodeclSet& private_ue_vars = task->get_private_ue_vars();
TL::Analysis::NodeclSet all_ue_vars(ue_vars.begin(), ue_vars.end());
all_ue_vars.insert(private_ue_vars.begin(), private_ue_vars.end());
const TL::Analysis::NodeclSet& killed_vars = task->get_killed_vars();
const TL::Analysis::NodeclSet& private_killed_vars = task->get_private_killed_vars();
TL::Analysis::NodeclSet all_killed_vars(killed_vars.begin(), killed_vars.end());
all_killed_vars.insert(private_killed_vars.begin(), private_killed_vars.end());
TL::Analysis::NodeclSet all_vars = all_ue_vars;
all_vars.insert(all_killed_vars.begin(), all_killed_vars.end());
const TL::Analysis::NodeclSet& used_addrs = task->get_used_addresses();
Nodecl::List tmp_dependency_vars;
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_DEP_IN))
tmp_dependency_vars.append(task_pragma_info.get_clause(NODECL_OPEN_M_P_DEP_IN).as<Nodecl::OpenMP::DepIn>().get_exprs().shallow_copy());
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_DEP_OUT))
tmp_dependency_vars.append(task_pragma_info.get_clause(NODECL_OPEN_M_P_DEP_OUT).as<Nodecl::OpenMP::DepOut>().get_exprs().shallow_copy());
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_DEP_INOUT))
tmp_dependency_vars.append(task_pragma_info.get_clause(NODECL_OPEN_M_P_DEP_INOUT).as<Nodecl::OpenMP::DepInout>().get_exprs().shallow_copy());
TL::Analysis::NodeclSet dependency_vars;
for(Nodecl::List::iterator it = tmp_dependency_vars.begin(); it != tmp_dependency_vars.end(); ++it)
{
if(it->is<Nodecl::Shaping>())
{
dependency_vars.insert(it->as<Nodecl::Shaping>().get_postfix());
const Nodecl::List& shapes = it->as<Nodecl::Shaping>().get_shape().as<Nodecl::List>();
for(Nodecl::List::iterator its = shapes.begin(); its != shapes.end(); ++its)
dependency_vars.insert(*its);
}
else
{
dependency_vars.insert(*it);
}
}
unsigned int n_unnecessarily_scoped_vars = 0;
std::string unnecessarily_scoped_vars = get_unnecessarily_scoped_vars(
task_scoped_vars, n_unnecessarily_scoped_vars,
all_vars, dependency_vars, used_addrs, task);
if (!unnecessarily_scoped_vars.empty())
{
unnecessarily_scoped_vars = unnecessarily_scoped_vars.substr(0, unnecessarily_scoped_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_unused_scope_message(n_unnecessarily_scoped_vars>1, task).c_str(),
unnecessarily_scoped_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __Unused, unnecessarily_scoped_vars);
}
unsigned int n_fp_dead_vars = 0;
std::string firstprivate_dead_vars = get_dead_vars(firstprivate_vars, n_fp_dead_vars, all_killed_vars, task);
if (!firstprivate_dead_vars.empty())
{
firstprivate_dead_vars = firstprivate_dead_vars.substr(0, firstprivate_dead_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_dead_vars_message( (n_fp_dead_vars>1), "firstprivate", task).c_str(), 
firstprivate_dead_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __FP_Dead, firstprivate_dead_vars);
}
unsigned int n_p_dead_vars = 0;
std::string private_dead_vars = get_dead_vars(private_vars, n_p_dead_vars, all_killed_vars, task);
if (!private_dead_vars.empty())
{
private_dead_vars = private_dead_vars.substr(0, private_dead_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_dead_vars_message( (n_p_dead_vars>1), "private", task).c_str(), 
private_dead_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __P_Dead, private_dead_vars);
}
unsigned int n_incoherent_private_vars = 0;
std::string incoherent_private_vars = get_incoherent_private_vars(private_vars, n_incoherent_private_vars, all_ue_vars, task);
if (!incoherent_private_vars.empty())
{
incoherent_private_vars = incoherent_private_vars.substr(0, incoherent_private_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_incoherent_private_message(n_incoherent_private_vars>1, task).c_str(),
incoherent_private_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __P_Incoherent, incoherent_private_vars);
}
unsigned int n_incoherent_firstprivate_vars = 0;
std::string incoherent_firstprivate_vars = get_incoherent_firstprivate_vars(
firstprivate_vars, n_incoherent_firstprivate_vars, all_ue_vars, dependency_vars, used_addrs, task);
if (!incoherent_firstprivate_vars.empty())
{
incoherent_firstprivate_vars = incoherent_firstprivate_vars.substr(0, incoherent_firstprivate_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_incoherent_firstprivate_message(n_incoherent_firstprivate_vars>1, task).c_str(),
incoherent_firstprivate_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __FP_Incoherent, incoherent_firstprivate_vars);
}
}
void check_task_incoherent_dependencies(TL::Analysis::Node* task)
{
Nodecl::List dep_in_vars, dep_out_vars;
TL::Analysis::PCFGPragmaInfo task_pragma_info = task->get_pragma_node_info( );
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_DEP_IN))
{
dep_in_vars = task_pragma_info.get_clause(NODECL_OPEN_M_P_DEP_IN).as<Nodecl::OpenMP::DepIn>().get_exprs().shallow_copy().as<Nodecl::List>();
}
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_DEP_OUT))
{
dep_out_vars = task_pragma_info.get_clause(NODECL_OPEN_M_P_DEP_OUT).as<Nodecl::OpenMP::DepOut>().get_exprs().shallow_copy().as<Nodecl::List>();
}
if (task_pragma_info.has_clause(NODECL_OPEN_M_P_DEP_INOUT))
{
Nodecl::List dep_inout_vars = task_pragma_info.get_clause(NODECL_OPEN_M_P_DEP_INOUT).as<Nodecl::OpenMP::DepInout>().get_exprs().shallow_copy().as<Nodecl::List>();
dep_in_vars.append(dep_inout_vars);
dep_out_vars.append(dep_inout_vars);
}
const TL::Analysis::NodeclSet& ue_vars = task->get_ue_vars();
const TL::Analysis::NodeclSet& killed_vars = task->get_killed_vars();
std::string incoherent_depin_vars;
std::string incoherent_depin_pointed_vars;
unsigned int n_incoherent_depin_vars = 0;
unsigned int n_incoherent_depin_pointed_vars = 0;
for(Nodecl::List::iterator it = dep_in_vars.begin(); it != dep_in_vars.end(); ++it)
{
if(it->is<Nodecl::Shaping>())
{   
const Nodecl::NodeclBase& var = it->as<Nodecl::Shaping>().get_postfix();
Nodecl::List enclosed = TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(var, ue_vars);
bool report_var = false;
if (enclosed.is_null())
{
if (TL::Analysis::Utils::nodecl_set_contains_pointed_nodecl(var, ue_vars).is_null())
report_var = true;
}
else
{
report_var = true;
for (Nodecl::List::const_iterator itt = enclosed.begin(); itt != enclosed.end(); ++itt)
{
const Nodecl::NodeclBase& e = itt->no_conv();
if (Nodecl::Utils::structurally_equal_nodecls(e, var, true)
|| !TL::Analysis::Utils::nodecl_set_contains_pointed_nodecl(e, ue_vars).is_null())
{
report_var = false;
break;
}
}
}
if (report_var)
{
incoherent_depin_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_in_var(*it);
n_incoherent_depin_vars++;
}
}
else
{   
const Nodecl::NodeclBase& var = *it;
if (!TL::Analysis::Utils::nodecl_set_contains_pointed_nodecl(var, ue_vars).is_null())
{
incoherent_depin_pointed_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_in_pointed_var(var);
n_incoherent_depin_pointed_vars++;
}
else if (!TL::Analysis::Utils::nodecl_set_contains_nodecl(var, ue_vars) &&
TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(var, ue_vars).is_null())
{
incoherent_depin_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_in_var(var);
n_incoherent_depin_vars++;
}
}
}
if (!incoherent_depin_vars.empty())
{
incoherent_depin_vars = incoherent_depin_vars.substr(0, incoherent_depin_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_incoherent_in_deps_message( (n_incoherent_depin_vars>1),  false, task).c_str(), 
incoherent_depin_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __IN_Incoherent, incoherent_depin_vars);
}
if (!incoherent_depin_pointed_vars.empty())
{
incoherent_depin_pointed_vars = incoherent_depin_pointed_vars.substr(0, incoherent_depin_pointed_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_incoherent_in_deps_message( (n_incoherent_depin_pointed_vars>1),  true, task).c_str(), 
incoherent_depin_pointed_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __IN_Pointed_Incoherent, incoherent_depin_pointed_vars);
}
std::string incoherent_depout_vars;
std::string incoherent_depout_pointed_vars;
unsigned int n_incoherent_depout_vars = 0;
unsigned int n_incoherent_depout_pointed_vars = 0;
for(Nodecl::List::iterator it = dep_out_vars.begin(); it != dep_out_vars.end(); ++it)
{
if(it->is<Nodecl::Shaping>())
{   
const Nodecl::NodeclBase& var = it->as<Nodecl::Shaping>().get_postfix();
Nodecl::List enclosed = TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(var, killed_vars);
bool report_var = false;
if (enclosed.is_null())
{
if (TL::Analysis::Utils::nodecl_set_contains_pointed_nodecl(var, killed_vars).is_null())
report_var = true;
}
else
{
report_var = true;
for (Nodecl::List::const_iterator itt = enclosed.begin(); itt != enclosed.end(); ++itt)
{
const Nodecl::NodeclBase& e = itt->no_conv();
if (Nodecl::Utils::structurally_equal_nodecls(var, e, true)
|| !TL::Analysis::Utils::nodecl_set_contains_pointed_nodecl(e, killed_vars).is_null())
{
report_var = false;
break;
}
}
}
if (report_var)
{
incoherent_depout_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_out_var(*it);
n_incoherent_depout_vars++;
}
}
else
{   
const Nodecl::NodeclBase& var = *it;
if(!TL::Analysis::Utils::nodecl_set_contains_pointed_nodecl(var, killed_vars).is_null())
{
incoherent_depout_pointed_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_out_pointed_var(var);
n_incoherent_depout_pointed_vars++;
}
else if(!TL::Analysis::Utils::nodecl_set_contains_nodecl(var, killed_vars) &&
TL::Analysis::Utils::nodecl_set_contains_enclosed_nodecl(var, killed_vars).is_null())
{
incoherent_depout_vars += it->prettyprint() + ", ";
task->add_correctness_incoherent_out_var(var);
n_incoherent_depout_vars++;
}
}
}
if (!incoherent_depout_vars.empty())
{
incoherent_depout_vars = incoherent_depout_vars.substr(0, incoherent_depout_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_incoherent_out_deps_message( (n_incoherent_depout_vars>1),  false, task).c_str(), 
incoherent_depout_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __OUT_Incoherent, incoherent_depout_vars);
}
if (!incoherent_depout_pointed_vars.empty())
{
incoherent_depout_pointed_vars = incoherent_depout_pointed_vars.substr(0, incoherent_depout_pointed_vars.size()-2);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_incoherent_out_deps_message( (n_incoherent_depout_pointed_vars>1),  true, task).c_str(), 
incoherent_depout_pointed_vars.c_str());
print_warn_to_file(task->get_graph_related_ast(), __OUT_Pointed_Incoherent, incoherent_depout_pointed_vars);
}
}
}
static void create_logs_file(std::string file_path)
{
if(!file_path.empty())
{
struct stat st;
if(stat(file_path.c_str(), &st) != 0)
{   
int old_mask = umask(0000);
int dot_directory = mkdir(file_path.c_str(), S_IRWXU|S_IRWXG|S_IRWXO);
umask(old_mask);
if(dot_directory != 0)
internal_error("An error occurred while creating the dot files directory in '%s'", file_path.c_str());
}
char* tmp_usr_name = getenv("USER");
usr_name = std::string(tmp_usr_name);
if(usr_name.empty())
usr_name = "undefined";
std::string date_str;
{
time_t t = time(NULL);
struct tm* tmp = localtime(&t);
if(tmp == NULL)
{
internal_error("localtime failed", 0);
}
char outstr[200];
if(strftime(outstr, sizeof(outstr), "%s", tmp) == 0)
{
internal_error("strftime failed", 0);
}
outstr[199] = '\0';
date_str = outstr;
}
char absolute_path[PATH_MAX+1];
char* path_ptr = realpath(file_path.c_str(), absolute_path);
ERROR_CONDITION(path_ptr == NULL, "Error retrieving the real path of path %s.\n", file_path.c_str());
snprintf(log_file_name, PATH_MAX, "%s/__correctness_%s_%lu_%s.log",
absolute_path,
usr_name.c_str(), (unsigned long)getppid(), date_str.c_str());
log_file_name[PATH_MAX-1] = '\0';
if (VERBOSE)
{
std::cerr << "OMP-LINT_ The correctness log files for this compilation will be stored in file: '" << log_file_name << "'" << std::endl;
}
int old_mask = umask(0022);
log_file = fopen(log_file_name, "a");
umask(old_mask);
if(log_file == NULL)
internal_error("Unable to open the file '%s' to store the correctness logs.", log_file_name);
}
}
static void execute_correctness_checks(TL::Analysis::ExtensibleGraph* graph)
{
if (VERBOSE)
std::cerr << "Correctness checks of PCFG '" << graph->get_name() << "'" << std::endl;
ObjectList<TL::Analysis::Node*> tasks = graph->get_tasks_list();
for (ObjectList<TL::Analysis::Node*>::iterator it = tasks.begin(); it != tasks.end(); it++)
{
TL::Analysis::Node* task = *it;
const Nodecl::NodeclBase& task_nodecl = task->get_graph_related_ast();
{
Nodecl::List local_vars;
if (task_is_locally_bound(task, local_vars).is_true() && 
task_only_synchronizes_in_enclosing_scopes(task, local_vars).is_false())
{
for(Nodecl::List::iterator itv = local_vars.begin(); itv != local_vars.end(); ++itv)
task->add_correctness_auto_storage_var(*itv);
std::string local_vars_str = get_nodecl_list_str(local_vars);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_auto_storage_message( local_vars.size()>1, task).c_str(), 
local_vars_str.c_str());
print_warn_to_file(task_nodecl, __SharedAutoStorage, local_vars_str);
}
}
{
TL::Analysis::NodeclTriboolMap race_vars;
std::map<TL::Analysis::Node*, VarToNodesMap> treated_scopes;
if (task_may_cause_race_condition(graph, task, treated_scopes, race_vars).is_true())
{
std::string race_vars_str = get_nodecl_map_str(race_vars);
warn_printf_at (task->get_graph_related_ast().get_locus(), get_race_message( race_vars.size()>1, task).c_str(),
race_vars_str.c_str());
print_warn_to_file(task_nodecl, __Race, race_vars_str);
}
}
{
check_task_incoherent_data_sharing(task);
}
{
check_task_incoherent_dependencies(task);
}
}
}
void launch_correctness(
const TL::Analysis::AnalysisBase& analysis,
std::string file_path)
{
create_logs_file(file_path);
const TL::ObjectList<TL::Analysis::ExtensibleGraph*>& extensible_graphs = analysis.get_pcfgs();
for (TL::ObjectList<TL::Analysis::ExtensibleGraph*>::const_iterator it = extensible_graphs.begin();
it != extensible_graphs.end(); ++it)
{
execute_correctness_checks(*it);
}
if(!file_path.empty())
fclose(log_file);
}
WritesVisitor::WritesVisitor( )
: _defined_vars( ), _define( false )
{}
ObjectList<Nodecl::NodeclBase> WritesVisitor::get_defined_symbols( )
{
return _defined_vars;
}
void WritesVisitor::clear( )
{
_defined_vars.clear( );
_define = false;
}
void WritesVisitor::visit_assignment( const Nodecl::NodeclBase& lhs, const Nodecl::NodeclBase& rhs )
{
_define = true;
walk( lhs );
_define = false;
walk( rhs );
}
void WritesVisitor::visit_xx_crement( const Nodecl::NodeclBase& rhs )
{
_define = true;
walk( rhs );
_define = false;
}
void WritesVisitor::visit( const Nodecl::AddAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::ArithmeticShrAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::ArraySubscript& n )
{
if( _define )
_defined_vars.insert( n );
walk( n.get_subscripts( ) );
}
void WritesVisitor::visit( const Nodecl::Assignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::BitwiseAndAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::BitwiseOrAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::BitwiseShlAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::BitwiseShrAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::BitwiseXorAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::ClassMemberAccess& n )
{
if( _define )
_defined_vars.insert( n );
}
void WritesVisitor::visit( const Nodecl::Dereference& n )
{
if( _define )
_defined_vars.insert( n );
}
void WritesVisitor::visit( const Nodecl::DivAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::MinusAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::ModAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::MulAssignment& n )
{
visit_assignment( n.get_lhs( ), n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::ObjectInit& n )
{
if( !n.get_symbol( ).get_value( ).is_null( ) )
{
Nodecl::Symbol n_sym = Nodecl::Symbol::make( n.get_symbol( ), n.get_locus( ) );
_defined_vars.insert( n_sym );
}
}
void WritesVisitor::visit( const Nodecl::Postdecrement& n )
{
visit_xx_crement( n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::Postincrement& n )
{
visit_xx_crement( n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::Predecrement& n )
{
visit_xx_crement( n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::Preincrement& n )
{
visit_xx_crement( n.get_rhs( ) );
}
void WritesVisitor::visit( const Nodecl::Reference& n )
{
if( _define )
_defined_vars.insert( n );
}
void WritesVisitor::visit( const Nodecl::Symbol& n )
{
if( _define )
_defined_vars.insert( n );
}
Lint::Lint()
: _disable_phase("0"), _correctness_log_path(""), _lint_deprecated_flag("")
{
set_phase_name("OpenMP Lint");
set_phase_description("This phase is able to detect some common pitfalls when using OpenMP");
register_parameter("disable-omp-lint",
"Disables this phase. You should not need this. If you do, then it is an error. Please fill a bug",
_disable_phase,
"0");
register_parameter("correctness_log_dir",
"Sets the path where correctness logs will be stored, in addition to showing them in the standard output",
_correctness_log_path,
"");
register_parameter("lint_deprecated_flag",
"Emits a warning as the \"--openmp-lint\" flag is deprecated",
_lint_deprecated_flag,
"0").connect(std::bind(&Lint::set_lint_deprecated_flag, this, std::placeholders::_1));
register_parameter("ompss_mode",
"Enables OmpSs semantics instead of OpenMP semantics",
_ompss_mode_str,
"0").connect(std::bind(&Lint::set_ompss_mode, this, std::placeholders::_1));
}
void Lint::run(TL::DTO& dto)
{
Nodecl::NodeclBase top_level = *std::static_pointer_cast<Nodecl::NodeclBase>(dto["nodecl"]);
if (_disable_phase == "0")
{
log_file_path = _correctness_log_path;
ompss_mode_enabled = _ompss_mode_enabled;
TL::Analysis::AnalysisBase analysis(ompss_mode_enabled);
if (VERBOSE)
{
std::cerr << "===========================================" << std::endl;
std::cerr << "OMP-LINT_ Executing analysis required for OpenMP/OmpSs correctness checking in file '" 
<< top_level.get_filename() << "'" << std::endl;
}
analysis.reaching_definitions(top_level,  true);
if (VERBOSE)
{
analysis.print_all_pcfg();
}
launch_correctness(analysis, log_file_path);
if (VERBOSE)
{
std::cerr << "===========================================" << std::endl;
}
}
}
void Lint::pre_run(TL::DTO& dto)
{}
void Lint::set_ompss_mode( const std::string& ompss_mode_str)
{
if( ompss_mode_str == "1")
_ompss_mode_enabled = true;
}
void Lint::set_lint_deprecated_flag(const std::string& lint_deprecated_flag_str)
{
if (lint_deprecated_flag_str == "1")
{
fprintf(stderr, "%s: parameter '--openmp-lint' deprecated. Use '--task-correctness' instead\n",
::compilation_process.exec_basename);
}
}
}
}
EXPORT_PHASE(TL::OpenMP::Lint)
