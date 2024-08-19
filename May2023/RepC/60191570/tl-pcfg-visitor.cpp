#include "cxx-cexpr.h"
#include "cxx-process.h"
#include "cxx-diagnostic.h"
#include "tl-analysis-utils.hpp"
#include "tl-pcfg-visitor.hpp"
#include <cassert>
namespace TL {
namespace Analysis {
std::map<Symbol, Nodecl::NodeclBase> analysis_asserted_decls;
PCFGVisitor::PCFGVisitor(std::string name, NBase nodecl)
: _utils(NULL), _pcfg(NULL), _asserted_funcs()
{
_utils = new PCFGVisitUtils();
_pcfg = new ExtensibleGraph(name, nodecl, _utils);
}
void PCFGVisitor::set_actual_pcfg(ExtensibleGraph* graph)
{
_pcfg = graph;
}
ExtensibleGraph* PCFGVisitor::parallel_control_flow_graph(
const NBase& n,
const std::map<Symbol, NBase>& asserted_funcs)
{
_asserted_funcs = asserted_funcs;
walk(n);
Node* pcfg_exit = _pcfg->_graph->get_graph_exit_node();
pcfg_exit->set_id(++_utils->_nid);
_pcfg->connect_nodes(_utils->_last_nodes, pcfg_exit);   
Node* returns_exit = pcfg_exit;
const ObjectList<Node*>& pcfg_exit_parents = pcfg_exit->get_parents();
if((pcfg_exit_parents.size() == 1) && (pcfg_exit_parents[0]->is_context_node()))
returns_exit = pcfg_exit_parents[0]->get_graph_exit_node();
for (ObjectList<Node*>::iterator it = _utils->_return_nodes.begin();
it != _utils->_return_nodes.end(); ++it)
{
Node* source = *it;
Node* outer = source->get_outer_node();
Node* exit = outer->get_graph_exit_node();
ObjectList<Node*> parents = exit->get_parents();
if (parents.empty())
{   
while (parents.empty())
{
_pcfg->connect_nodes(source, exit);
_pcfg->disconnect_nodes(outer, outer->get_children());
source = outer;
outer = outer->get_outer_node();
if (outer == NULL)
goto next_it;
exit = outer->get_graph_exit_node();
parents = exit->get_parents();
}
}
_pcfg->connect_nodes(source, returns_exit);
next_it:    ;
}
_utils->_return_nodes.clear();
return _pcfg;
}
void PCFGVisitor::compute_catch_parents(Node* node)
{
while(!node->is_visited())
{
node->set_visited(true);
NodeType n_type = node->get_type();
if(n_type == __Graph)
compute_catch_parents(node->get_graph_entry_node());
else if(n_type == __Exit)
return;
else if(n_type != __Entry && n_type != __UnclassifiedNode && n_type != __Break)
_utils->_tryblock_nodes.back()->_handler_parents.append(node);
ObjectList<Edge*> exit_edges = node->get_exit_edges();
for(ObjectList<Edge*>::iterator it = exit_edges.begin(); it != exit_edges.end(); it++)
compute_catch_parents((*it)->get_target());
}
}
ObjectList<Node*> PCFGVisitor::get_first_nodes(Node* actual_node)
{
ObjectList<Edge*> actual_entries = actual_node->get_entry_edges();
ObjectList<Node*> actual_parents;
if(actual_entries.empty())
{
if(actual_node->is_entry_node())
return ObjectList<Node*>();
else
return ObjectList<Node*>(1, actual_node);
}
else
{
for(ObjectList<Edge*>::iterator it = actual_entries.begin(); it != actual_entries.end(); ++it)
{
ObjectList<Node*> parents = get_first_nodes((*it)->get_source());
actual_parents.insert(parents);
}
}
return actual_parents;
}
Node* PCFGVisitor::merge_nodes(NBase n, ObjectList<Node*> nodes_l)
{
Node* result;
NodeType ntype;
if(n.is<Nodecl::FunctionCall>() || n.is<Nodecl::VirtualFunctionCall>())
{
ntype = (_utils->_is_vector ? __VectorFunctionCall : __FunctionCall);
}
else if(n.is<Nodecl::LabeledStatement>())
{
if(_utils->_is_vector)
internal_error("Merging vector node with labeled statement is not yet implemented\n", 0);
ntype = __Labeled;
}
else
{
ntype = (_utils->_is_vector ? __VectorNormal : __Normal);
}
if(nodes_l.size() > 1
|| ((nodes_l.size() == 1) && (nodes_l[0]->get_type() == __Graph)))
{   
bool need_graph = false;
for(ObjectList<Node*>::iterator it = nodes_l.begin(); it != nodes_l.end(); ++it)
{
if((*it)->get_type() == __Graph)
{
need_graph = true;
break;
}
}
if(need_graph)
{
bool found;
result = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __SplitStmt);
Node* entry = result->get_graph_entry_node();
ObjectList<Node*> graph_parents;
ObjectList<int> list_pos_to_erase;
int i = 0;
for(ObjectList<Node*>::iterator it = nodes_l.begin(); it != nodes_l.end(); ++it)
{
found = false;
ObjectList<Node*> actual_parents = (*it)->get_parents();
ObjectList<Node*>::iterator iit;
for(iit = nodes_l.begin(); iit != nodes_l.end(); ++iit)
{
if(actual_parents.contains(*iit))
{
found = true;
break;
}
}
if(!found)
{
graph_parents.append((*it)->get_parents());
ObjectList<Node*> aux = (*it)->get_parents();
for(ObjectList<Node*>::iterator iit2 = aux.begin(); iit2 != aux.end(); ++iit2)
{
(*iit2)->erase_exit_edge(*it);
(*it)->erase_entry_edge(*iit2);
}
if((*it)->get_type() != __Graph)
{
list_pos_to_erase.append(i);
delete (*it);
}
else
{
_pcfg->connect_nodes(entry, *it);
}
}
i++;
}
if(!graph_parents.empty())
{
_pcfg->connect_nodes(graph_parents, result);
}
for(ObjectList<int>::reverse_iterator it = list_pos_to_erase.rbegin();
it != list_pos_to_erase.rend(); ++it)
{
nodes_l.erase(nodes_l.begin() + (*it));
}
Node* merged_node = new Node(_utils->_nid, ntype, result, n);
ObjectList<Node*> merged_parents;
for(ObjectList<Node*>::iterator it = nodes_l.begin(); it != nodes_l.end(); ++it)
{
found = false;
ObjectList<Node*> actual_children = (*it)->get_children();
for(ObjectList<Node*>::iterator iit = nodes_l.begin(); iit != nodes_l.end(); ++iit)
{
if(actual_children.contains(*iit))
{
found = true;
break;
}
}
if(!found)
{
merged_parents.append(*it);
}
(*it)->set_outer_node(result);
}
_pcfg->connect_nodes(merged_parents, merged_node);
Node* graph_exit = result->get_graph_exit_node();
graph_exit->set_id(++_utils->_nid);
_pcfg->connect_nodes(merged_node, graph_exit);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, result);
}
else
{
for(ObjectList<Node*>::iterator it = nodes_l.begin(); it != nodes_l.end(); ++it)
{
ObjectList<Node*> aux = (*it)->get_parents();
if(!aux.empty())
{
ObjectList<NBase> stmts = (*it)->get_statements();
std::string stmts_str = "";
for(ObjectList<NBase>::iterator it2 = stmts.begin(); it2 != stmts.end(); ++it2)
{
stmts_str += it2->prettyprint() + "\n";
}
internal_error("Deleting node (%d) of type %s that has '%d' parents. \n" \
"This type of node shouldn't be already connected.",
(*it)->get_id(), (*it)->get_type_as_string().c_str(), aux.size());
}
delete (*it);
}
result = new Node(_utils->_nid, ntype, _utils->_outer_nodes.top(), n);
}
}
else
{
result = new Node(_utils->_nid, ntype, _utils->_outer_nodes.top(), n);
}
return result;
}
Node* PCFGVisitor::merge_nodes(NBase n, Node* first, Node* second)
{
ObjectList<Node*> previous_nodes;
previous_nodes.append(first);
if(second != NULL)
{   
previous_nodes.append(second);
}
return merge_nodes(n, previous_nodes);
}
ObjectList<Node*> PCFGVisitor::visit_barrier(const NBase& n)
{
Node* barrier_graph = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpBarrierGraph);
_pcfg->connect_nodes(_utils->_last_nodes, barrier_graph);
Node* barrier_entry = barrier_graph->get_graph_entry_node();
Node* barrier_exit = barrier_graph->get_graph_exit_node();
Node* flush_1 = new Node(_utils->_nid, __OmpFlush, barrier_graph);
_pcfg->connect_nodes(barrier_entry, flush_1);
Node* barrier = new Node(_utils->_nid, __OmpBarrier, barrier_graph);
_pcfg->connect_nodes(flush_1, barrier);
Node* flush_2 = new Node(_utils->_nid, __OmpFlush, barrier_graph);
_pcfg->connect_nodes(barrier, flush_2);
barrier_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(flush_2, barrier_exit);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, barrier_graph);
return ObjectList<Node*>(1, barrier_graph);
}
ObjectList<Node*> PCFGVisitor::visit_binary_node(const NBase& n,
const NBase& lhs,
const NBase& rhs)
{
bool is_vector = _utils->_is_vector;
Node* left = walk(lhs)[0];
_utils->_is_vector = is_vector;
Node* right = walk(rhs)[0];
_utils->_is_vector = is_vector;
return ObjectList<Node*>(1, merge_nodes(n, left, right));
}
ObjectList<Node*> PCFGVisitor::visit_case_or_default(const NBase& case_stmt,
const Nodecl::List& case_val)
{
Node* case_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), case_stmt, __SwitchCase);
if(_utils->_last_nodes.size() == 1 && _utils->_last_nodes[0]->is_entry_node() &&
_utils->_last_nodes[0]->get_outer_node()->is_context_node())
{
ERROR_CONDITION(_utils->_switch_nodes.top()->_condition != NULL,
"When visiting the first case of a Switch statement, "
"the PCFG node containing the condition of the Switch should be NULL, "
"because it shall be set to the Entry node of the Context node created inside the Switch.\n", 0);
_utils->_switch_nodes.top()->set_condition(_utils->_last_nodes[0]);
}
_pcfg->connect_nodes(_utils->_last_nodes, case_node);
Edge* e = _pcfg->connect_nodes(_utils->_switch_nodes.top()->_condition, case_node, __Case);
ERROR_CONDITION(case_val.size()>1, "Case statement '%s' with more than one value per case is not yet supported\n",
case_stmt.prettyprint().c_str());
if(case_val.size()==1)
e->add_label(case_val[0]);
Node* entry_node = case_node->get_graph_entry_node();
Node* exit_node = case_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, entry_node);
_utils->_break_nodes.push(exit_node);
ObjectList<Node*> case_stmts = walk(case_stmt);
_utils->_break_nodes.pop();
exit_node->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, exit_node);
if(case_stmts.empty() ||
((case_stmts.size() == 1) && !case_stmts[0]->is_break_node() &&
(!case_stmts[0]->is_context_node() ||
(case_stmts[0]->is_context_node() &&
!case_stmts[0]->get_graph_exit_node()->get_parents().empty()))))
{
_utils->_last_nodes = ObjectList<Node*>(1, case_node);
}
else
_pcfg->connect_nodes(exit_node, _utils->_switch_nodes.top()->_exit);
_utils->_outer_nodes.pop();
return ObjectList<Node*>(1, case_node);
}
template <typename T>
ObjectList<Node*> PCFGVisitor::visit_conditional_expression(const T& n)
{
GraphType n_type = (_utils->_is_vector ? __VectorCondExpr : __CondExpr);
Node* cond_expr_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, n_type);
Node* entry_node = cond_expr_node->get_graph_entry_node();
bool is_vector = _utils->_is_vector;
Node* condition_node = walk(n.get_condition())[0];
_utils->_is_vector = is_vector;
_pcfg->connect_nodes(entry_node, condition_node);
ObjectList<Node*> exit_parents;
Node* true_node = walk(n.get_true())[0];
_utils->_is_vector = is_vector;
_pcfg->connect_nodes(condition_node, true_node);
exit_parents.append(true_node);
Node* false_node = walk(n.get_false())[0];
_utils->_is_vector = is_vector;
_pcfg->connect_nodes(condition_node, false_node);
exit_parents.append(false_node);
Node* exit_node = cond_expr_node->get_graph_exit_node();
exit_node->set_id(++(_utils->_nid));
_pcfg->connect_nodes(exit_parents, exit_node);
_utils->_outer_nodes.pop();
return ObjectList<Node*>(1, cond_expr_node);
}
template <typename T>
ObjectList<Node*> PCFGVisitor::visit_function_call(const T& n)
{
_pcfg->add_func_call_symbol(n.get_called().get_symbol());
Node* func_graph_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n,
(_utils->_is_vector ? __VectorFunctionCallGraph :
__FunctionCallGraph));
if(!_utils->_last_nodes.empty())
{   
_pcfg->connect_nodes(_utils->_last_nodes, func_graph_node);
}
_utils->_last_nodes.clear();
Node* func_node;
bool is_vector = _utils->_is_vector;
Nodecl::List args = n.get_arguments().template as<Nodecl::List>();
ObjectList<Node*> arguments_l = walk(args);
_utils->_is_vector = is_vector;
if(!arguments_l.empty())
func_node = merge_nodes(n, arguments_l);
else
func_node = new Node(_utils->_nid, (_utils->_is_vector ? __VectorFunctionCall : __FunctionCall),
func_graph_node, n);
_pcfg->connect_nodes(func_graph_node->get_graph_entry_node(), func_node);
Node* graph_exit = func_graph_node->get_graph_exit_node();
graph_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(func_node, graph_exit);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, func_graph_node);
return ObjectList<Node*>(1, func_graph_node);
}
ObjectList<Node*> PCFGVisitor::visit_literal_node(const NBase& n)
{
NodeType n_type = (_utils->_is_vector ? __VectorNormal : __Normal);
Node* basic_node = new Node(_utils->_nid, n_type, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, basic_node);
}
ObjectList<Node*> PCFGVisitor::visit_taskwait(const NBase& n)
{
Node* taskwait_node = new Node(_utils->_nid, __OmpTaskwait, _utils->_outer_nodes.top(), n);
_pcfg->connect_nodes(_utils->_last_nodes, taskwait_node);
_utils->_last_nodes = ObjectList<Node*>(1, taskwait_node);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit_taskwait_on(const NBase& n)
{
Node* taskwait_node = new Node(_utils->_nid, __OmpWaitonDeps, _utils->_outer_nodes.top(), n);
_pcfg->connect_nodes(_utils->_last_nodes, taskwait_node);
_utils->_last_nodes = ObjectList<Node*>(1, taskwait_node);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit_unary_node(const NBase& n,
const NBase& rhs)
{
bool is_vector = _utils->_is_vector;
Node* right = walk(rhs)[0];
_utils->_is_vector = is_vector;
return ObjectList<Node*>(1, merge_nodes(n, right, NULL));
}
ObjectList<Node*> PCFGVisitor::visit_vector_binary_node(const NBase& n,
const NBase& lhs,
const NBase& rhs)
{
_utils->_is_vector = true;
ObjectList<Node*> result = visit_binary_node(n, lhs, rhs);
_utils->_is_vector = false;
return result;
}
template <typename T>
ObjectList<Node*> PCFGVisitor::visit_vector_function_call(const T& n)
{
NBase called_func = n.get_function_call();
if(!called_func.is<Nodecl::FunctionCall>())
{
internal_error("Unexpected nodecl type '%s' as function call member of a vector function call\n",
ast_print_node_type(called_func.get_kind()));
}
_utils->_is_vector = true;
ObjectList<Node*> vector_func_node_l = visit_function_call(called_func.as<Nodecl::FunctionCall>());
_utils->_is_vector = false;
vector_func_node_l[0]->set_graph_label(n);
return vector_func_node_l;
}
ObjectList<Node*> PCFGVisitor::visit_vector_unary_node(const NBase& n,
const NBase& rhs)
{
_utils->_is_vector = true;
ObjectList<Node*> result = visit_unary_node(n, rhs);
_utils->_is_vector = false;
return result;
}
ObjectList<Node*> PCFGVisitor::visit_vector_memory_func(const NBase& n, char mem_access_type)
{
NodeType n_type;
if(mem_access_type == '1')
n_type = __VectorLoad;
else if(mem_access_type == '2')
n_type = __VectorGather;
else if(mem_access_type == '3')
n_type = __VectorStore;
else if(mem_access_type == '4')
n_type = __VectorScatter;
else
internal_error("Unexpected type '%c' of vector memory access. Expecting types from 1 to 2\n", mem_access_type);
Node* vector_mem_node = new Node(_utils->_nid, n_type, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, vector_mem_node);
}
ObjectList<Node*> PCFGVisitor::unhandled_node(const NBase& n)
{
if(VERBOSE) {
WARNING_MESSAGE("Unhandled node of type '%s' while PCFG construction.\n '%s' ",
ast_print_node_type(n.get_kind()), n.prettyprint().c_str());
} else {
WARNING_MESSAGE("Unhandled node of type '%s' while PCFG construction.\n", ast_print_node_type(n.get_kind()));
}
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Add& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::AddAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Alignof& n)
{
return visit_unary_node(n, n.get_align_type());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Assert& n)
{
ObjectList<Node*> stmts = walk(n.get_statements());
ERROR_CONDITION((stmts.size() != 1),
"The expected number of nodes returned while traversing "\
"the Analysis::Assert statements is one, but %s returned", stmts.size());
_utils->_assert_nodes.push(stmts[0]);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
walk(n.get_environment());
_utils->_assert_nodes.pop();
_utils->_pragma_nodes.pop();
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::AssertDecl& n)
{
analysis_asserted_decls[n.get_symbol()] = n.get_environment();
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::AutoScope::Firstprivate& n)
{
_utils->_assert_nodes.top()->add_assert_auto_sc_firstprivate_var(n.get_scoped_variables().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::AutoScope::Private& n)
{
_utils->_assert_nodes.top()->add_assert_auto_sc_private_var(n.get_scoped_variables().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::AutoScope::Shared& n)
{
_utils->_assert_nodes.top()->add_assert_auto_sc_shared_var(n.get_scoped_variables().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::AutoStorage& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_auto_storage_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::Dead& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_dead_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::IncoherentFp& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_incoherent_fp_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::IncoherentIn& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_incoherent_in_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::IncoherentInPointed& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_incoherent_in_pointed_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::IncoherentOut& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_incoherent_out_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::IncoherentOutPointed& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_incoherent_out_pointed_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::IncoherentP& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_incoherent_p_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Correctness::Race& n)
{
_utils->_assert_nodes.top()->add_assert_correctness_race_var(n.get_correctness_vars().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Dead& n)
{
_utils->_assert_nodes.top()->add_assert_dead_var(n.get_dead_exprs().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Defined& n)
{
_utils->_assert_nodes.top()->add_assert_killed_var(n.get_defined_exprs().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::InductionVariable& n)
{
_utils->_assert_nodes.top()->add_assert_induction_variables(n.get_induction_variables().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::LiveIn& n)
{
_utils->_assert_nodes.top()->add_assert_live_in_var(n.get_live_in_exprs().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::LiveOut& n)
{
_utils->_assert_nodes.top()->add_assert_live_out_var(n.get_live_out_exprs().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Range& n)
{
_utils->_assert_nodes.top()->add_assert_ranges(n.get_range_variables().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::ReachingDefinitionIn& n)
{
_utils->_assert_nodes.top()->add_assert_reaching_definitions_in(n.get_reaching_definitions_in().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::ReachingDefinitionOut& n)
{
_utils->_assert_nodes.top()->add_assert_reaching_definitions_out(n.get_reaching_definitions_out().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::Undefined& n)
{
_utils->_assert_nodes.top()->add_assert_undefined_behaviour_var(n.get_undefined_exprs().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Analysis::UpperExposed& n)
{
_utils->_assert_nodes.top()->add_assert_ue_var(n.get_upper_exposed_exprs().as<Nodecl::List>());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ArithmeticShr& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ArithmeticShrAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ArraySubscript& n)
{
bool is_vector = _utils->_is_vector;
ObjectList<Node*> subscripted = walk(n.get_subscripted());
_utils->_is_vector = is_vector;
ObjectList<Node*> subscripts = walk(n.get_subscripts());
_utils->_is_vector = is_vector;
ObjectList<Node*> nodes = subscripted;
nodes.insert(subscripts);
return ObjectList<Node*>(1, merge_nodes(n, nodes));
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Assignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseAnd& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseAndAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseNot& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseOr& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseOrAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseShl& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseShlAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseShr& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseShrAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseXor& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BitwiseXorAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BooleanLiteral& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::BreakStatement& n)
{
Node* break_node;
if(_utils->_last_nodes.empty())
break_node = _pcfg->append_new_child_to_parent(_utils->_switch_nodes.top()->_condition, n, __Break);
else
break_node = _pcfg->append_new_child_to_parent(_utils->_last_nodes, n, __Break);
_pcfg->connect_nodes(break_node, _utils->_break_nodes.top());
_utils->_last_nodes.clear();
return ObjectList<Node*>(1, break_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::CaseStatement& n)
{
return visit_case_or_default(n.get_statement(), n.get_case().as<Nodecl::List>());
}
ObjectList<Node*> PCFGVisitor::visit( const Nodecl::CatchHandler& n )
{
PCFGTryBlock* current_tryblock = _utils->_tryblock_nodes.back();
current_tryblock->_nhandlers++;
_utils->_last_nodes = current_tryblock->_handler_parents;
ObjectList<Node*> catchs = walk(n.get_statement());
current_tryblock->_handler_exits.append(catchs[0]);
NBase label = n.get_name();
if(label.is_null())
{
const char* s = "...";
label = Nodecl::StringLiteral::make(Type(get_literal_string_type(strlen(s)+1,
get_char_type())),
const_value_make_string(s, strlen(s)));
}
for(ObjectList<Node*>::iterator it = current_tryblock->_handler_parents.begin();
it != current_tryblock->_handler_parents.end(); ++it)
{
Edge* catch_edge = (*it)->get_exit_edge(catchs[0]);
if(catch_edge != NULL)
{
catch_edge->set_catch_edge();
catch_edge->add_label(label);
}
}
return catchs;
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ClassMemberAccess& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_member());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Comma& n)
{
ObjectList<Node*> comma_nodes;
bool is_vector = _utils->_is_vector;
comma_nodes.append(walk(n.get_lhs()));
_utils->_is_vector = is_vector;
comma_nodes.append(walk(n.get_rhs()));
_utils->_is_vector = is_vector;
return comma_nodes;
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ComplexLiteral& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::CompoundExpression& n)
{
return walk(n.get_nest());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::CompoundStatement& n)
{
return walk(n.get_statements());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Concat& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ConditionalExpression& n)
{
return visit_conditional_expression(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Context& n)
{
_utils->_context_nodecl.push(n);
Node* context_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __Context);
_pcfg->connect_nodes(_utils->_last_nodes, context_node);
Node* context_entry = context_node->get_graph_entry_node();
Node* context_exit = context_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, context_entry);
walk(n.get_in_context());
context_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, context_exit);
_utils->_context_nodecl.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, context_node);
return ObjectList<Node*>(1, context_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ContinueStatement& n)
{
Node* continue_node = _pcfg->append_new_child_to_parent(_utils->_last_nodes, n, __Continue);
_pcfg->connect_nodes(continue_node, _utils->_continue_nodes.top());
_utils->_last_nodes.clear();
return ObjectList<Node*>(1, continue_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Conversion& n)
{
return walk(n.get_nest());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::CxxDef& n)
{   
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::CxxDecl& n)
{   
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::CxxUsingNamespace& n)
{
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::DefaultStatement& n)
{
return visit_case_or_default(n.get_statement(), Nodecl::List());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Delete& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::DeleteArray& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Dereference& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Different& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Div& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::DivAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::DoStatement& n)
{
Node* do_graph_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __LoopDoWhile);
_pcfg->connect_nodes(_utils->_last_nodes, do_graph_node);
Node* do_exit = do_graph_node->get_graph_exit_node();
_utils->_last_nodes.clear();
Node* condition_node = walk(n.get_condition())[0];
do_graph_node->set_condition_node(condition_node);
_utils->_last_nodes = ObjectList<Node*>(1, do_graph_node->get_graph_entry_node());
_utils->_continue_nodes.push(condition_node);
_utils->_break_nodes.push(do_exit);
ObjectList<Node*> stmts = walk(n.get_statement());
_utils->_continue_nodes.pop();
_utils->_break_nodes.pop();
_pcfg->connect_nodes(_utils->_last_nodes, condition_node);
if(!stmts.empty())
{
_pcfg->connect_nodes(condition_node, stmts[0], __TrueEdge, NBase::null(), 
false, true);
}
do_exit->set_id(++(_utils->_nid));
do_exit->set_outer_node(_utils->_outer_nodes.top());
_pcfg->connect_nodes(condition_node, do_exit, __FalseEdge);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, do_graph_node);
return ObjectList<Node*>(1, condition_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::EmptyStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Equal& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ExpressionStatement& n)
{
ObjectList<Node*> expr_last_nodes = _utils->_last_nodes;
ObjectList<Node*> expression_nodes = walk(n.get_nest());
if(expression_nodes.size() > 0)
{
Node* last_node;
if(expression_nodes.size() == 1)
last_node = expression_nodes[0];
else        
last_node = merge_nodes(n, expression_nodes);
ObjectList<Node*> expr_first_nodes = get_first_nodes(last_node);
for(ObjectList<Node*>::iterator it = expr_first_nodes.begin();
it != expr_first_nodes.end(); ++it)
{
_pcfg->clear_visits(*it);
}
if(!expr_last_nodes.empty())
{   
int n_connects = expr_first_nodes.size() * expr_last_nodes.size();
if(n_connects != 0)
{
_pcfg->connect_nodes(expr_last_nodes, expr_first_nodes);
}
}
if(!_utils->_last_nodes.empty())
_utils->_last_nodes = ObjectList<Node*>(1, last_node);
}
else
{
internal_error("Parsing the expression '%s' 0 nodes has been returned, and they must be one or more\n",
codegen_to_str(n.get_internal_nodecl(),
nodecl_retrieve_context(n.get_internal_nodecl())));
}
return expression_nodes;
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FieldDesignator& n)
{
ObjectList<Node*> field = walk(n.get_field());
ObjectList<Node*> next = walk(n.get_next());
Node* result;
if(n.get_next().is<Nodecl::StructuredValue>())
{   
result = next[0];
ERROR_CONDITION(next.size()!=1,
"More that one node created traversing the 'next' member of a FieldDesignator\n", 0);
Node* node_to_modify;
if(next[0]->is_graph_node())
{
ObjectList<Node*> exit_parents = next[0]->get_graph_exit_node()->get_parents();
ERROR_CONDITION(exit_parents.size()!=1,
"More than one parent found for the exit node of an split_node ", 0);
node_to_modify = exit_parents[0];
}
else
{
node_to_modify = next[0];
}
ObjectList<NBase> stmts = node_to_modify->get_statements();
ERROR_CONDITION(stmts.size()!=1, "More than one statement created for the 'next' member of a FieldDesignator", 0);
if(stmts[0].is<Nodecl::FieldDesignator>())
{
Nodecl::FieldDesignator fd = stmts[0].as<Nodecl::FieldDesignator>();
Type t = fd.get_field().get_symbol().get_type();
Nodecl::ClassMemberAccess new_lhs =
Nodecl::ClassMemberAccess::make(n.get_field().shallow_copy(), fd.get_field().shallow_copy(),
NBase::null(),
t, n.get_locus());
Nodecl::Assignment new_assign =
Nodecl::Assignment::make(new_lhs, fd.get_next().shallow_copy(), t, n.get_locus());
node_to_modify->set_statements(ObjectList<NBase>(1, new_assign));
}
else
{
internal_error("Unexpected node '%s' when FieldDesignator expected\n",
ast_print_node_type(stmts[0].get_kind()));
}
}
else
{   
result = merge_nodes(n, field[0], next[0]);
}
return ObjectList<Node*>(1, result);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FloatingLiteral& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ForStatement& n)
{
ObjectList<Node*> actual_last_nodes = _utils->_last_nodes;
walk(n.get_loop_header());
Node* init = _utils->_nested_loop_nodes.top()->_init;
Node* cond = _utils->_nested_loop_nodes.top()->_cond;
Node* next = _utils->_nested_loop_nodes.top()->_next;
_utils->_last_nodes = actual_last_nodes;
if(init != NULL)
{
_pcfg->connect_nodes(_utils->_last_nodes, init);
Node* last_init_node = init;
ObjectList<Node*> init_children = last_init_node->get_children();
while(!init_children.empty())
{
ERROR_CONDITION(init_children.size() != 1,
"A LoopControl init can generate more than one node, but no branches are allowed", 0);
last_init_node = init_children[0];
init_children = last_init_node->get_children();
}
_utils->_last_nodes = ObjectList<Node*>(1, last_init_node);
}
Node* for_graph_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __LoopFor);
_pcfg->connect_nodes(_utils->_last_nodes, for_graph_node);
Node* entry_node = for_graph_node->get_graph_entry_node();
if(cond != NULL)
{
cond->set_outer_node(for_graph_node);
_pcfg->connect_nodes(entry_node, cond);
_utils->_last_nodes = ObjectList<Node*>(1, cond);
for_graph_node->set_condition_node(cond);
}
else
{
_utils->_last_nodes = ObjectList<Node*>(1, entry_node);
}
Node* exit_node = for_graph_node->get_graph_exit_node();
if(next != NULL)
_utils->_continue_nodes.push(next);
else
_utils->_continue_nodes.push(exit_node);
_utils->_break_nodes.push(exit_node);
walk(n.get_statement());
_utils->_continue_nodes.pop();
_utils->_break_nodes.pop();
exit_node->set_id(++(_utils->_nid));
EdgeType aux_etype = __Always;
if(cond != NULL)
{
ObjectList<Edge*> exit_edges = cond->get_exit_edges();
if(!exit_edges.empty())
{
bool all_tasks = true;
ObjectList<Edge*>::iterator it = exit_edges.begin();
while(all_tasks && it != exit_edges.end())
{
if(!(*it)->is_task_edge())
{
all_tasks = false;
}
(*it)->set_true_edge();
++it;
}
if(all_tasks)
aux_etype = __TrueEdge;
}
else
{   
aux_etype = __TrueEdge;
}
_pcfg->connect_nodes(cond, exit_node, __FalseEdge);
}
if(next != NULL)
{
Node* first_next = next;
Node* last_next = next;
last_next->set_outer_node(for_graph_node);
while(!last_next->get_exit_edges().empty())
{
last_next = last_next->get_children()[0];
last_next->set_outer_node(for_graph_node);
}
_pcfg->connect_nodes(_utils->_last_nodes, first_next, ObjectList<EdgeType>(_utils->_last_nodes.size(), aux_etype));
if(cond != NULL)
{   
_pcfg->connect_nodes(last_next, cond, __Always, NBase::null(), 
false, true);
}
else
{   
_pcfg->connect_nodes(last_next, exit_node);
}
}
else
{
if(cond != NULL)
{   
if((_utils->_last_nodes.size() == 1) && (_utils->_last_nodes[0] == cond))
_pcfg->connect_nodes(cond, cond, __TrueEdge);
else
_pcfg->connect_nodes(_utils->_last_nodes, cond);
}
}
if(exit_node->get_parents().empty())
{   
_pcfg->connect_nodes(_utils->_last_nodes, exit_node);
}
_utils->_nested_loop_nodes.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, for_graph_node);
return ObjectList<Node*>(1, for_graph_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FunctionCall& n)
{
return visit_function_call(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FunctionCode& n)
{
Symbol func_sym(n.get_symbol());
_pcfg->_function_sym = func_sym;
Node* func_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __FunctionCode);
_pcfg->connect_nodes(_utils->_last_nodes, func_node);
Node* func_entry = func_node->get_graph_entry_node();
Node* func_exit = func_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, func_entry);
walk(n.get_statements());
func_exit->set_id(++_utils->_nid);
_pcfg->connect_nodes(_utils->_last_nodes, func_exit);
if(_asserted_funcs.find(func_sym) != _asserted_funcs.end())
{   
_utils->_assert_nodes.push(func_node);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
walk(_asserted_funcs[func_sym]);
_utils->_assert_nodes.pop();
_utils->_pragma_nodes.pop();
}
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, func_node);
return ObjectList<Node*>(1, func_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::GccAsmDefinition& n)
{
Node* asm_def_graph_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __AsmDef);
_pcfg->connect_nodes(_utils->_last_nodes, asm_def_graph_node);
Node* entry_node = asm_def_graph_node->get_graph_entry_node();
Node* exit_node = asm_def_graph_node->get_graph_exit_node();
Nodecl::Text text = Nodecl::Text::make(n.get_text());
Node* text_node = new Node(_utils->_nid, __Normal, _utils->_outer_nodes.top(), text);
text_node->set_asm_info(ASM_DEF_TEXT);
_pcfg->connect_nodes(entry_node, text_node);
_utils->_last_nodes = ObjectList<Node*>(1, text_node);
ObjectList<Node*> op0 = walk(n.get_operands0());
if(!op0.empty())
{
op0[0]->set_asm_info(ASM_DEF_OUTPUT_OPS);
}
ObjectList<Node*> op1 = walk(n.get_operands1());
if (!op1.empty())
{
op1[0]->set_asm_info(ASM_DEF_INPUT_OPS);
}
ObjectList<Node*> op2 = walk(n.get_operands2());
if(!op2.empty())
{
op2[0]->set_asm_info(ASM_DEF_CLOBBERED_REGS);
}
exit_node->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, exit_node);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, asm_def_graph_node);
return ObjectList<Node*>(1, asm_def_graph_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::GccAsmOperand& n)
{
Node* asm_op_node = new Node(_utils->_nid, __AsmOp, _utils->_outer_nodes.top(), n);
_pcfg->connect_nodes(_utils->_last_nodes, asm_op_node);
_utils->_last_nodes = ObjectList<Node*>(1, asm_op_node);
return ObjectList<Node*>(1, asm_op_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::GccBuiltinVaArg& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::GotoStatement& n)
{
Node* goto_node = _pcfg->append_new_child_to_parent(_utils->_last_nodes, n, __Goto);
goto_node->set_label(n.get_symbol());
_pcfg->connect_nodes(_utils->_last_nodes, goto_node);
ObjectList<Node*>::iterator it;
for(it = _utils->_labeled_nodes.begin(); it != _utils->_labeled_nodes.end(); ++it)
{
if((*it)->get_label() == n.get_symbol())
{   
Nodecl::Symbol s = Nodecl::Symbol::make(n.get_symbol());
_pcfg->connect_nodes(goto_node, *it, __GotoEdge, s, false, true);
break;
}
}
if(it == _utils->_labeled_nodes.end())
{
_utils->_goto_nodes.append(goto_node);
}
_utils->_last_nodes.clear();
return ObjectList<Node*>(1, goto_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::GreaterOrEqualThan& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::GreaterThan& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::IfElseStatement& n)
{
Node* if_else_graph_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __IfElse);
_pcfg->connect_nodes(_utils->_last_nodes, if_else_graph_node);
Node* if_else_exit = if_else_graph_node->get_graph_exit_node();
_utils->_last_nodes.clear();
Node* cond_node = walk(n.get_condition())[0];
_pcfg->connect_nodes(if_else_graph_node->get_graph_entry_node(), cond_node);
_utils->_last_nodes = ObjectList<Node*>(1, cond_node);
if_else_graph_node->set_condition_node(cond_node);
ObjectList<Node*> then_node_l = walk(n.get_then());
if(!cond_node->get_exit_edges().empty())
{
ObjectList<Edge*> exit_edges = cond_node->get_exit_edges();
bool all_tasks_then = true;
for(ObjectList<Edge*>::iterator it = exit_edges.begin(); it != exit_edges.end(); ++it)
{   
(*it)->set_true_edge();
if(!(*it)->is_task_edge())
all_tasks_then = false;
}
_pcfg->connect_nodes(_utils->_last_nodes, if_else_exit);
_utils->_last_nodes = ObjectList<Node*>(1, cond_node);
ObjectList<Node*> else_node_l = walk(n.get_else());
bool all_tasks_else = true;
unsigned int false_edge_it = exit_edges.size();
exit_edges = cond_node->get_exit_edges();
for( ; false_edge_it < cond_node->get_exit_edges().size(); ++false_edge_it)
{
exit_edges[false_edge_it]->set_false_edge();
if(!exit_edges[false_edge_it]->is_task_edge())
all_tasks_else = false;
}
if(!else_node_l.empty())
_pcfg->connect_nodes(_utils->_last_nodes, if_else_exit);
else
_pcfg->connect_nodes(_utils->_last_nodes, if_else_exit, ObjectList<EdgeType>(_utils->_last_nodes.size(), __FalseEdge));
if((all_tasks_then && all_tasks_else) || cond_node->get_exit_edges().empty())
{
_pcfg->connect_nodes(cond_node, if_else_exit);
}
else if(all_tasks_then)
{
_pcfg->connect_nodes(cond_node, if_else_exit, __TrueEdge);
}
else if(all_tasks_else)
{
_pcfg->connect_nodes(cond_node, if_else_exit, __FalseEdge);
}
}
else
{
_pcfg->connect_nodes(cond_node, if_else_exit);
}
if_else_exit->set_id(++(_utils->_nid));
if_else_exit->set_outer_node(_utils->_outer_nodes.top());
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, if_else_graph_node);
return ObjectList<Node*>(1, if_else_graph_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::IndexDesignator& n)
{   
return walk(n.get_next());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::IntegerLiteral& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::IntelAssume& n)
{
Node* basic_node = new Node(_utils->_nid, __Builtin, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, basic_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::IntelAssumeAligned& n)
{
Node* basic_node = new Node(_utils->_nid, __Builtin, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, basic_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::LabeledStatement& n)
{
bool is_vector = _utils->_is_vector;
Node* labeled_node = walk(n.get_statement())[0];
_utils->_is_vector = is_vector;
labeled_node->set_type(__Labeled);
labeled_node->set_label(n.get_symbol());
for(ObjectList<Node*>::iterator it = _utils->_goto_nodes.begin();
it != _utils->_goto_nodes.end(); ++it)
{
if((*it)->get_label() == n.get_symbol())
{   
const char* s = n.get_symbol().get_name().c_str();
int slen = strlen(s);
NBase label = Nodecl::StringLiteral::make(
Type(get_literal_string_type(slen+1, get_char_type())),
const_value_make_string(s, slen));
_pcfg->connect_nodes(*it, labeled_node, __GotoEdge, label);
break;
}
}
_utils->_labeled_nodes.append(labeled_node);
if(labeled_node->get_exit_edges().empty())
_utils->_last_nodes = ObjectList<Node*>(1, labeled_node);
return ObjectList<Node*>(1, labeled_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::LogicalAnd& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::LogicalNot& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::LogicalOr& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit_loop_control(
const NBase& init,
const NBase& cond,
const NBase& next)
{
PCFGLoopControl* current_loop_ctrl = new PCFGLoopControl();
_utils->_last_nodes.clear();
ObjectList<Node*> init_node_l = walk(init);
if(init_node_l.empty())
{   
current_loop_ctrl->_init = NULL;
}
else
{
current_loop_ctrl->_init = init_node_l[0];
}
_utils->_last_nodes.clear();
ObjectList<Node*> cond_node_l = walk(cond);
if(cond_node_l.empty())
{   
current_loop_ctrl->_cond = NULL;
}
else
{
current_loop_ctrl->_cond = cond_node_l[0];
}
_utils->_last_nodes.clear();
ObjectList<Node*> next_node_l = walk(next);
if(next_node_l.empty())
{
current_loop_ctrl->_next = NULL;
}
else
{
if(next_node_l.size() > 1)
{   
ObjectList<Node*>::iterator it = next_node_l.begin();
ObjectList<Node*>::iterator itt = it; ++itt;
for(; itt != next_node_l.end(); ++it, ++itt)
{
_pcfg->connect_nodes(*it, *itt);
}
}
current_loop_ctrl->_next = next_node_l[0];
}
_utils->_nested_loop_nodes.push(current_loop_ctrl);
return ObjectList<Node*>();   
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::LoopControl& n)
{
return visit_loop_control(n.get_init(), n.get_cond(), n.get_next());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::LowerOrEqualThan& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::LowerThan& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::MaskLiteral& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Minus& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::MinusAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Mod& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ModAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Mul& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::MulAssignment& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Neg& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::New& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ObjectInit& n)
{
if(_pcfg == NULL)
{   
return ObjectList<Node*>();
}
else
{
Nodecl::Symbol n_sym = Nodecl::Symbol::make(n.get_symbol(), n.get_locus());
Type n_type = n.get_symbol().get_type();
if(n_type.is_aggregate() || n_type.is_class() || n_type.is_array())
{   
ObjectList<Node*> init_exprs = walk(n.get_symbol().get_value());
if (init_exprs.empty())
{   
return ObjectList<Node*>();
}
else
{
bool unnamed_member_initialization = false;
for(ObjectList<Node*>::iterator it = init_exprs.begin(); it != init_exprs.end(); ++it)
{
if((*it)->is_graph_node())
{
ObjectList<Node*> exit_parents = (*it)->get_graph_exit_node()->get_parents();
ERROR_CONDITION(exit_parents.size() != 1,
"More than one parent found for the exit node of an split_node ", 0);
Node* exit_parent = exit_parents[0];
if (exit_parent->is_graph_node())
{   
unnamed_member_initialization = true;
continue;
}
else
{
ObjectList<NBase> stmts = exit_parent->get_statements();
ERROR_CONDITION(stmts.size() != 1, "More than one statement found in the last node of an split_node", 0);
if(stmts[0].is<Nodecl::Assignment>())
{   
Nodecl::Assignment ass = stmts[0].as<Nodecl::Assignment>();
Nodecl::ClassMemberAccess new_lhs =
Nodecl::ClassMemberAccess::make(n_sym, ass.get_lhs().shallow_copy(),
NBase::null(),
ass.get_type(), n.get_locus());
NBase new_assign =
Nodecl::Assignment::make(new_lhs, ass.get_rhs().shallow_copy(), ass.get_type(), n.get_locus());
exit_parent->set_statements(ObjectList<NBase>(1, new_assign));
}
else if(stmts[0].is<Nodecl::FieldDesignator>())
{   
Nodecl::FieldDesignator fd = stmts[0].as<Nodecl::FieldDesignator>();
Type t = fd.get_field().get_symbol().get_type();
Nodecl::ClassMemberAccess new_lhs =
Nodecl::ClassMemberAccess::make(n_sym, fd.get_field().shallow_copy(),
NBase::null(),
t, n.get_locus());
NBase new_assign =
Nodecl::Assignment::make(new_lhs, fd.get_next().shallow_copy(), t, n.get_locus());
exit_parent->set_statements(ObjectList<NBase>(1, new_assign));
}
else
{   
unnamed_member_initialization = true;
continue;
}
}
_utils->_last_nodes = ObjectList<Node*>(1, *it);
}
else
{
ObjectList<NBase> it_expr = (*it)->get_statements();
ERROR_CONDITION(it_expr.size() != 1,
"More than one statement created for an structured value initialization\n", 0);
NBase init_expr = it_expr[0];
NBase it_init;
if(init_expr.is<Nodecl::Assignment>())
{   
Nodecl::Assignment ass = init_expr.as<Nodecl::Assignment>();
Nodecl::ClassMemberAccess new_lhs =
Nodecl::ClassMemberAccess::make(n_sym, ass.get_lhs().shallow_copy(),
NBase::null(),
ass.get_type(), n.get_locus());
it_init = Nodecl::Assignment::make(new_lhs, ass.get_rhs().shallow_copy(), ass.get_type(), n.get_locus());
}
else if(init_expr.is<Nodecl::FieldDesignator>())
{   
Nodecl::FieldDesignator fd = init_expr.as<Nodecl::FieldDesignator>();
Type t = fd.get_field().get_symbol().get_type();
Nodecl::ClassMemberAccess new_lhs =
Nodecl::ClassMemberAccess::make(n_sym, fd.get_field().shallow_copy(),
NBase::null(),
t, n.get_locus());
it_init = Nodecl::Assignment::make(new_lhs, fd.get_next().shallow_copy(), t, n.get_locus());
}
else
{   
unnamed_member_initialization = true;
continue;
}
Node* it_init_node = new Node( _utils->_nid, __Normal, _utils->_outer_nodes.top( ), it_init );
_pcfg->connect_nodes( _utils->_last_nodes, it_init_node );
_utils->_last_nodes = ObjectList<Node*>( 1, it_init_node );
}
}
if(unnamed_member_initialization)
{   
Node* it_init_node = new Node(_utils->_nid, __Normal, _utils->_outer_nodes.top(), n);
_pcfg->connect_nodes(_utils->_last_nodes, it_init_node);
_utils->_last_nodes = ObjectList<Node*>(1, it_init_node);
}
}
}
else
{
ObjectList<Node*> init_last_nodes = _utils->_last_nodes;
ObjectList<Node*> init_expr = walk(n.get_symbol().get_value());
if(init_expr.empty())
{   
return ObjectList<Node*>();
}
else
{
ERROR_CONDITION(init_expr.size() != 1,
"An ObjectInit of a variables which is neither a class, nor an aggregate nor an array "\
"must have at most one node generated for the initializing expression, but %d found", init_expr.size());
Node* init_node = merge_nodes(n, init_expr[0], NULL);
if(!init_last_nodes.empty())     
_pcfg->connect_nodes(init_last_nodes, init_node);
_utils->_last_nodes = ObjectList<Node*>(1, init_node);
}
}
}
return _utils->_last_nodes;
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Offset& n)
{
return visit_binary_node(n, n.get_base(), n.get_offset());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Offsetof& n)
{
return visit_binary_node(n, n.get_offset_type(), n.get_designator());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::Alloca& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::CopyIn& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::CopyInout& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::CopyOut& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::Cost& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepCommutative& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepConcurrent& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepInPrivate& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepReduction& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepWeakCommutative& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepWeakInout& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepWeakIn& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepWeakOut& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::DepWeakReduction& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::Lint& n)
{
Node* lint_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpssLint);
_pcfg->connect_nodes(_utils->_last_nodes, lint_node);
Node* lint_entry = lint_node->get_graph_entry_node();
Node* lint_exit = lint_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, lint_entry);
walk(n.get_statements());
lint_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, lint_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(lint_entry, lint_exit));
walk(n.get_environment());
lint_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, lint_node);
return ObjectList<Node*>(1, lint_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::LintAlloc& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::LintFree& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::LintVerified& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::SharedAndAlloca& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::Target& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::TaskCall& n)
{
Node* task_creation = new Node(_utils->_nid, __OmpTaskCreation, _utils->_outer_nodes.top());
_pcfg->connect_nodes(_utils->_last_nodes, task_creation);
Node* task_node = _pcfg->create_graph_node(_pcfg->_graph, n, __OmpTask, _utils->_context_nodecl.top());
const char* s = "Create";
int slen = strlen(s);
NBase label = Nodecl::StringLiteral::make(
Type(get_literal_string_type(slen+1, get_char_type())), const_value_make_string(s, slen));
_pcfg->connect_nodes(task_creation, task_node, __Always, label,  true);
Node* task_entry = task_node->get_graph_entry_node();
Node* task_exit = task_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, task_entry);
walk(n.get_call());
task_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, task_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(task_entry, task_exit));
walk(n.get_site_environment());
task_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, task_creation);
_pcfg->_task_nodes_l.insert(task_node);
return ObjectList<Node*>(1, task_creation);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::TaskExpression& n)
{
walk(n.get_task_calls());
return walk(n.get_join_task());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::TaskLabel& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OmpSs::WeakReduction& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Aligned& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Atomic& n)
{
Node* atomic_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpAtomic);
_pcfg->connect_nodes(_utils->_last_nodes, atomic_node);
Node* atomic_entry = atomic_node->get_graph_entry_node();
Node* atomic_exit = atomic_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, atomic_entry);
walk(n.get_statements());
atomic_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, atomic_exit);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(atomic_entry, atomic_exit));
walk(n.get_environment());
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, atomic_node);
return ObjectList<Node*>(1, atomic_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Auto& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
_utils->_environ_entry_exit.top().first->get_outer_node()->set_auto_scoping_enabled();
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::BarrierAtEnd& n)
{
Node* environ_exit = _utils->_environ_entry_exit.top().second;
ObjectList<Node*> exit_parents = environ_exit->get_parents();
_pcfg->disconnect_nodes(exit_parents, environ_exit);
ObjectList<Node*> actual_last_nodes = _utils->_last_nodes;
_utils->_last_nodes = exit_parents;
visit_barrier(n);
_pcfg->connect_nodes(_utils->_last_nodes[0], environ_exit);
_utils->_last_nodes = actual_last_nodes;
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::BarrierFull& n)
{
return visit_barrier(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::BarrierSignal& n)
{
WARNING_MESSAGE("BarrierSignal not yet implemented. Ignoring nodecl", 0);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::BarrierWait& n)
{
WARNING_MESSAGE("BarrierWait not yet implemented. Ignoring nodecl", 0);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Overlap& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::CombinedWithParallel& n)
{
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Critical& n)
{
Node* critical_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpCritical);
_pcfg->connect_nodes(_utils->_last_nodes, critical_node);
Node* critical_entry = critical_node->get_graph_entry_node();
Node* critical_exit = critical_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, critical_entry);
walk(n.get_statements());
critical_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, critical_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(critical_entry, critical_exit));
walk(n.get_environment());
critical_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, critical_node);
return ObjectList<Node*>(1, critical_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::CriticalName& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::DepIn& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::DepInout& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::DepOut& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Device& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Final& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Firstprivate& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::FirstLastprivate& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::FlushAtEntry& n)
{
Node* environ_entry = _utils->_environ_entry_exit.top().first;
ObjectList<Node*> entry_children = environ_entry->get_children();
_pcfg->disconnect_nodes(environ_entry, entry_children);
ObjectList<Node*> actual_last_nodes = _utils->_last_nodes;
_utils->_last_nodes = ObjectList<Node*>(1, environ_entry);
Node* entry_flush = _pcfg->create_flush_node(_utils->_outer_nodes.top());
_pcfg->connect_nodes(entry_flush, entry_children);
environ_entry = entry_flush;
_utils->_last_nodes = actual_last_nodes;
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::FlushAtExit& n)
{
Node* environ_exit = _utils->_environ_entry_exit.top().second;
ObjectList<Node*> exit_parents = environ_exit->get_parents();
_pcfg->disconnect_nodes(exit_parents, environ_exit);
ObjectList<Node*> actual_last_nodes = _utils->_last_nodes;
_utils->_last_nodes = exit_parents;
Node* exit_flush = _pcfg->create_flush_node(_utils->_outer_nodes.top());
_pcfg->connect_nodes(exit_flush, environ_exit);
_utils->_last_nodes = actual_last_nodes;
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::FlushMemory& n)
{
return ObjectList<Node*>(1, _pcfg->create_flush_node(_utils->_outer_nodes.top(), n.get_expressions()));
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::For& n)
{
Node* for_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpLoop);
_pcfg->connect_nodes(_utils->_last_nodes, for_node);
Node* for_entry = for_node->get_graph_entry_node();
Node* for_exit = for_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, for_entry);
walk(n.get_loop());
for_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, for_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(for_entry, for_exit));
walk(n.get_environment());
for_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, for_node);
return ObjectList<Node*>(1, for_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::ForAppendix& n)
{
Node* for_app_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpForAppendix);
_pcfg->connect_nodes(_utils->_last_nodes, for_app_node);
Node* for_app_entry = for_app_node->get_graph_entry_node();
Node* for_app_exit = for_app_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, for_app_entry);
walk(n.get_loop());
walk(n.get_prependix());
walk(n.get_appendix());
for_app_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, for_app_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(for_app_entry, for_app_exit));
walk(n.get_environment());
for_app_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, for_app_node);
return ObjectList<Node*>(1, for_app_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::FunctionTaskParsingContext& n)
{
return walk(n.get_context());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::If& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Lastprivate& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Linear& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::MapFrom& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::MapTo& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::MapToFrom& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Mask& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Master& n)
{
Node* master_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpMaster);
_pcfg->connect_nodes(_utils->_last_nodes, master_node);
_utils->_last_nodes = ObjectList<Node*>(1, master_node->get_graph_entry_node());
walk(n.get_statements());
Node* master_exit = master_node->get_graph_exit_node();
master_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, master_exit);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, master_node);
return ObjectList<Node*>(1, master_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::NoMask& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Nontemporal& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Parallel& n)
{
Node* parallel_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpParallel);
_pcfg->connect_nodes(_utils->_last_nodes, parallel_node);
Node* parallel_entry = parallel_node->get_graph_entry_node();
Node* parallel_exit = parallel_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, parallel_entry);
walk(n.get_statements());
parallel_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, parallel_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(parallel_entry, parallel_exit));
walk(n.get_environment());
parallel_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, parallel_node);
return ObjectList<Node*>(1, parallel_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::ParallelSimdFor& n)
{
Node* simd_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpSimdParallelFor);
_pcfg->connect_nodes(_utils->_last_nodes, simd_node);
Node* simd_entry = simd_node->get_graph_entry_node();
Node* simd_exit = simd_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, simd_entry);
walk(n.get_statement());
simd_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, simd_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(simd_entry, simd_exit));
walk(n.get_environment());
simd_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, simd_node);
return ObjectList<Node*>(1, simd_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Priority& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Prefetch& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Private& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit( const Nodecl::OpenMP::PrivateInit& n )
{
_utils->_pragma_nodes.top( )._clauses.append(n);
return ObjectList<Node*>( );
}
ObjectList<Node*> PCFGVisitor::visit( const Nodecl::OpenMP::Reduction& n )
{
walk(n.get_reductions());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::ReductionItem& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Schedule& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Section& n)
{
ObjectList<Node*> section_last_nodes = _utils->_last_nodes;
Node* section_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpSection);
_pcfg->connect_nodes(_utils->_last_nodes, section_node);
_utils->_last_nodes = ObjectList<Node*>(1, section_node->get_graph_entry_node());
walk (n.get_statements());
Node* section_exit = section_node->get_graph_exit_node();
section_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, section_exit);
_utils->_outer_nodes.pop();
_utils->_section_nodes.top().append(section_node);
_utils->_last_nodes = section_last_nodes;
return ObjectList<Node*>(1, section_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Sections& n)
{
Node* sections_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpSections);
_pcfg->connect_nodes(_utils->_last_nodes, sections_node);
Node* sections_entry = sections_node->get_graph_entry_node();
Node* sections_exit = sections_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, sections_entry);
ObjectList<Node*> section_nodes;
_utils->_section_nodes.push(section_nodes);
walk(n.get_sections());
_utils->_last_nodes = _utils->_section_nodes.top();
_utils->_section_nodes.pop();
sections_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, sections_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(sections_entry, sections_exit));
walk(n.get_environment());
sections_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, sections_node);
return ObjectList<Node*>(1, sections_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Shared& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Simd& n)
{
Node* simd_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpSimd);
_pcfg->connect_nodes(_utils->_last_nodes, simd_node);
Node* simd_entry = simd_node->get_graph_entry_node();
Node* simd_exit = simd_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, simd_entry);
walk(n.get_statement());
simd_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, simd_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(simd_entry, simd_exit));
walk(n.get_environment());
simd_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, simd_node);
return ObjectList<Node*>(1, simd_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::SimdFor& n)
{
Node* simd_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpSimdFor);
_pcfg->connect_nodes(_utils->_last_nodes, simd_node);
Node* simd_entry = simd_node->get_graph_entry_node();
Node* simd_exit = simd_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, simd_entry);
walk(n.get_openmp_for());
simd_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, simd_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(simd_entry, simd_exit));
walk(n.get_environment());
simd_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, simd_node);
return ObjectList<Node*>(1, simd_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::SimdFunction& n)
{
Node* simd_function_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpSimdFunction);
_pcfg->connect_nodes(_utils->_last_nodes, simd_function_node);
Node* simd_function_entry = simd_function_node->get_graph_entry_node();
Node* simd_function_exit = simd_function_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, simd_function_entry);
walk(n.get_statement());
simd_function_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, simd_function_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(simd_function_entry, simd_function_exit));
walk(n.get_environment());
simd_function_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, simd_function_node);
return ObjectList<Node*>(1, simd_function_node);
}
ObjectList<Node*> PCFGVisitor::visit( const Nodecl::OpenMP::SimdReduction& n )
{
walk(n.get_reductions());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Single& n)
{
Node* single_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpSingle);
_pcfg->connect_nodes(_utils->_last_nodes, single_node);
Node* single_entry = single_node->get_graph_entry_node();
Node* single_exit = single_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, single_entry);
walk(n.get_statements());
single_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, single_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(single_entry, single_exit));
walk(n.get_environment());
single_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, single_node);
return ObjectList<Node*>(1, single_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Suitable& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Target& n)
{
Node* last_outer = _utils->_outer_nodes.top();
Node* target_node = _pcfg->create_graph_node(
_pcfg->_graph, n,
__OmpAsyncTarget, _utils->_context_nodecl.top());
Node* target_entry = target_node->get_graph_entry_node();
Node* target_exit = target_node->get_graph_exit_node();
Nodecl::List environ = n.get_environment().as<Nodecl::List>();
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(target_entry, target_exit));
walk(environ);
target_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
if (target_node->get_pragma_node_info().has_clause(NODECL_OPEN_M_P_TARGET_TASK_UNDEFERRED))
{
target_node->set_graph_type(__OmpSyncTarget);
_pcfg->connect_nodes(_utils->_last_nodes, target_node);
target_node->set_outer_node(last_outer);
_utils->_last_nodes = ObjectList<Node*>(1, target_entry);
walk(n.get_statements());
target_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, target_exit);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, target_node);
return ObjectList<Node*>(1, target_node);
}
else
{
_utils->_outer_nodes.pop();
Node* task_creation = new Node(_utils->_nid, __OmpTaskCreation, _utils->_outer_nodes.top());
_pcfg->connect_nodes(_utils->_last_nodes, task_creation);
const char* s = "Create";
int slen = strlen(s);
NBase label = Nodecl::StringLiteral::make(
Type(get_literal_string_type(slen+1, get_char_type())), const_value_make_string(s, slen));
_pcfg->connect_nodes(task_creation, target_node, __Always, label,  true);
_utils->_outer_nodes.push(target_node);     
_utils->_last_nodes = ObjectList<Node*>(1, target_entry);
walk(n.get_statements());
target_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, target_exit);
_utils->_outer_nodes.pop();
_pcfg->_task_nodes_l.insert(target_node);
_utils->_last_nodes = ObjectList<Node*>(1, task_creation);
return ObjectList<Node*>(1, task_creation);
}
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::TargetTaskUndeferred& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Task& n)
{
Node* task_creation = new Node(_utils->_nid, __OmpTaskCreation, _utils->_outer_nodes.top());
_pcfg->connect_nodes(_utils->_last_nodes, task_creation);
Node* task_node = _pcfg->create_graph_node(_pcfg->_graph, n, __OmpTask, _utils->_context_nodecl.top());
const char* s = "Create";
int slen = strlen(s);
NBase label = Nodecl::StringLiteral::make(
Type(get_literal_string_type(slen+1, get_char_type())), const_value_make_string(s, slen));
_pcfg->connect_nodes(task_creation, task_node, __Always, label,  true);
Node* task_entry = task_node->get_graph_entry_node();
Node* task_exit = task_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, task_entry);
walk(n.get_statements());
task_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, task_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(task_entry, task_exit));
walk(n.get_environment());
task_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_pcfg->_task_nodes_l.insert(task_node);
_utils->_last_nodes = ObjectList<Node*>(1, task_creation);
return ObjectList<Node*>(1, task_creation);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Taskwait& n)
{
struct DependencesVisitor : Nodecl::ExhaustiveVisitor<void>
{
bool has_dependences;
DependencesVisitor() : has_dependences(false) {}
void visit(const Nodecl::OpenMP::DepIn& n)
{
has_dependences = true;
}
void visit(const Nodecl::OpenMP::DepOut& n)
{
has_dependences = true;
}
void visit(const Nodecl::OpenMP::DepInout& n)
{
has_dependences = true;
}
void visit(const Nodecl::OmpSs::DepCommutative& n)
{
error_printf_at(n.get_locus(),
"commutative dependences are not supported on the taskwait construct\n");
}
void visit(const Nodecl::OmpSs::DepConcurrent& n)
{
error_printf_at(n.get_locus(),
"concurrent dependences are not supported on the taskwait construct\n");
}
};
Nodecl::List env = n.get_environment().as<Nodecl::List>();
DependencesVisitor visitor;
visitor.walk(env);
if (visitor.has_dependences)
return visit_taskwait_on(n);
else
return visit_taskwait(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Uniform& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Unroll& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Untied& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::VectorLengthFor& n)
{
_utils->_pragma_nodes.top()._clauses.append(n);
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::OpenMP::Workshare& n)
{
Node* single_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __OmpWorkshare);
_pcfg->connect_nodes(_utils->_last_nodes, single_node);
Node* single_entry = single_node->get_graph_entry_node();
Node* single_exit = single_node->get_graph_exit_node();
_utils->_last_nodes = ObjectList<Node*>(1, single_entry);
walk(n.get_statements());
single_exit->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, single_exit);
PCFGPragmaInfo current_pragma;
_utils->_pragma_nodes.push(current_pragma);
_utils->_environ_entry_exit.push(std::pair<Node*, Node*>(single_entry, single_exit));
walk(n.get_environment());
single_node->set_pragma_node_info(_utils->_pragma_nodes.top());
_utils->_pragma_nodes.pop();
_utils->_environ_entry_exit.pop();
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, single_node);
return ObjectList<Node*>(1, single_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ParenthesizedExpression& n)
{
ObjectList<Node*> current_last_nodes = _utils->_last_nodes;
bool is_vector = _utils->_is_vector;
ObjectList<Node*> expression_nodes = walk(n.get_nest());
_utils->_is_vector = is_vector;
Node* parenthesized_node = merge_nodes(n, expression_nodes);
return ObjectList<Node*>(1, parenthesized_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Plus& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::PointerToMember& n)
{
Node* basic_node = new Node(_utils->_nid, __Normal, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, basic_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Postdecrement& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Postincrement& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Power& n)
{
return visit_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::PragmaContext& n)
{
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::PragmaCustomDirective& n)
{
if(VERBOSE)
WARNING_MESSAGE("Ignoring PragmaCustomDirective \n'%s'", n.get_pragma_line().prettyprint().c_str());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::PragmaCustomStatement& n)
{
if(VERBOSE)
WARNING_MESSAGE("Ignoring PragmaCustomStatement '%s' but visiting its statements.",
n.get_pragma_line().prettyprint().c_str());
return walk(n.get_statements());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Predecrement& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Preincrement& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Range& n)
{
ObjectList<Node*> lower = walk(n.get_lower());
ObjectList<Node*> upper = walk(n.get_upper());
ObjectList<Node*> stride = walk(n.get_stride());
Node* merged_limits = merge_nodes(n, lower[0], upper[0]);
Node* merged = merge_nodes(n, merged_limits, stride[0]);
_utils->_last_nodes = ObjectList<Node*>(1, merged);
return ObjectList<Node*>(1, merged);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::RangeLoopControl& n)
{
abort();
Nodecl::Symbol induction_var = n.get_induction_variable().as<Nodecl::Symbol>();
NBase lower = n.get_lower();
NBase upper = n.get_upper();
NBase step = n.get_step();
if (step.is_null())
step = const_value_to_nodecl(const_value_get_signed_int(1));
TL::Symbol induction_sym = induction_var.get_symbol();
TL::Type sym_ref_type = induction_sym.get_type();
if (!sym_ref_type.is_any_reference())
sym_ref_type = sym_ref_type.get_lvalue_reference_to();
Nodecl::Symbol induction_var_ref = Nodecl::Symbol::make(induction_sym, induction_var.get_locus());
induction_var_ref.set_type(sym_ref_type);
NBase fake_init =
Nodecl::Assignment::make(
induction_var.shallow_copy(),
lower.shallow_copy(),
sym_ref_type);
NBase fake_cond;
if (step.is_constant())
{
const_value_t* c = step.get_constant();
if (const_value_is_negative(c))
{
fake_cond = Nodecl::GreaterOrEqualThan::make(
induction_var.shallow_copy(),
upper.shallow_copy(),
get_bool_type());
}
else
{
fake_cond = Nodecl::LowerOrEqualThan::make(
induction_var.shallow_copy(),
upper.shallow_copy(),
get_bool_type());
}
}
else
{
fake_cond = Nodecl::LowerOrEqualThan::make(
Nodecl::Mul::make(
induction_var.shallow_copy(),
step.shallow_copy(),
sym_ref_type.no_ref()),
Nodecl::Mul::make(
upper.shallow_copy(),
step.shallow_copy(),
sym_ref_type.no_ref()),
get_bool_type());
}
NBase fake_next =
Nodecl::Assignment::make(
induction_var.shallow_copy(),
Nodecl::Add::make(
induction_var.shallow_copy(),
step.shallow_copy(),
sym_ref_type.no_ref()),
sym_ref_type);
return visit_loop_control(fake_init, fake_cond, fake_next);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Reference& n)
{
return visit_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::ReturnStatement& n)
{
ObjectList<Node*> return_last_nodes = _utils->_last_nodes;
ObjectList<Node*> returned_value = walk(n.get_value());
Node* return_node = merge_nodes(n, returned_value);
if (return_node->is_graph_node())
{   
ObjectList<Node*> return_node_exit_parents
= return_node->get_graph_exit_node()->get_parents();
ERROR_CONDITION(return_node_exit_parents.size() > 1,
"One node expected but %d found.\n",
return_node_exit_parents.size());
Node* real_return_node = return_node_exit_parents[0];
real_return_node->set_type(__Return);
}
else
{
return_node->set_type(__Return);
}
_pcfg->connect_nodes(return_last_nodes, return_node);
_utils->_last_nodes.clear();
_utils->_return_nodes.append(return_node);
return ObjectList<Node*>(1, return_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Sizeof& n)
{
return visit_unary_node(n, n.get_size_type());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::StringLiteral& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::StructuredValue& n)
{
return walk(n.get_items());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::SwitchStatement& n)
{
Node* switch_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __Switch);
_pcfg->connect_nodes(_utils->_last_nodes, switch_node);
Node* entry_node = switch_node->get_graph_entry_node();
Node* exit_node = switch_node->get_graph_exit_node();
ObjectList<Node*> cond_last_nodes = _utils->_last_nodes;
ObjectList<Node*> cond_node_l = walk(n.get_switch());
Node* cond = cond_node_l[0];
_pcfg->connect_nodes(entry_node, cond);
switch_node->set_condition_node(cond);
_utils->_last_nodes = ObjectList<Node*>(1, cond);
_utils->_switch_nodes.push(new PCFGSwitch(NULL, exit_node)); 
_utils->_break_nodes.push(exit_node);
walk(n.get_statement());
_utils->_break_nodes.pop();
_utils->_switch_nodes.pop();
ObjectList<Node*> cond_children = cond->get_children();
Node* switch_ctx = NULL;
for(ObjectList<Node*>::iterator it = cond_children.begin(); it != cond_children.end(); ++it)
if((*it)->is_context_node())
{
switch_ctx = *it;
break;
}
ERROR_CONDITION(switch_ctx == NULL, "A switch condition node must have a context node as child, "\
"but %d does not have one", cond->get_id());
Node* ctx_entry = switch_ctx->get_graph_entry_node();
for(ObjectList<Node*>::iterator it = cond_children.begin(); it != cond_children.end(); ++it)
if(*it != switch_ctx)
{
Edge* e = ExtensibleGraph::get_edge_between_nodes(cond, *it);
_pcfg->disconnect_nodes(cond, *it);
_pcfg->connect_nodes(ctx_entry, *it, e->get_type(), e->get_label());
}
Node* ctx_exit = switch_ctx->get_graph_exit_node();
ObjectList<Node*> cases_exits = exit_node->get_parents();
for(ObjectList<Node*>::iterator it = cases_exits.begin(); it != cases_exits.end(); ++it)
{
Edge* e = ExtensibleGraph::get_edge_between_nodes(*it, exit_node);
_pcfg->disconnect_nodes(*it, exit_node);
_pcfg->connect_nodes(*it, ctx_exit, e->get_type(), e->get_label());
}
exit_node->set_id(++(_utils->_nid));
_pcfg->connect_nodes(_utils->_last_nodes, exit_node);
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, switch_node);
return ObjectList<Node*>(1, switch_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Symbol& n)
{
Scope s_sc = n.get_symbol().get_scope();
if (!s_sc.scope_is_enclosed_by(_pcfg->_sc) && !n.get_symbol().is_function())
_pcfg->_global_vars.insert(n);
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Text& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Throw& n)
{
ObjectList<Node*> right = walk(n.get_rhs());
Node* throw_node;
if(right.empty())
{   
throw_node = _pcfg->append_new_child_to_parent(_utils->_last_nodes, n);
}
else
{   
throw_node = merge_nodes(n, right[0], NULL);
_pcfg->connect_nodes(_utils->_last_nodes, throw_node);
}
if(!_utils->_tryblock_nodes.empty())
{
for(ObjectList<PCFGTryBlock*>::reverse_iterator it = _utils->_tryblock_nodes.rbegin();
it != _utils->_tryblock_nodes.rend(); ++it)
{
(*it)->_handler_parents.append(throw_node);
}
}
_pcfg->connect_nodes(throw_node, _pcfg->_graph->get_graph_exit_node());
_utils->_last_nodes.clear();
return ObjectList<Node*>(1, throw_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::TryBlock& n)
{
PCFGTryBlock* new_try_block = new PCFGTryBlock();
_utils->_tryblock_nodes.append(new_try_block);
ObjectList<Node*> try_parents = _utils->_last_nodes;
ObjectList<Node*> try_stmts = walk(n.get_statement());
Node* first_try_node = try_parents[0]->get_exit_edges()[0]->get_target();
compute_catch_parents(first_try_node);
_pcfg->clear_visits(first_try_node);
ObjectList<Node*> handlers_l = walk(n.get_catch_handlers());
ObjectList<Node*> ellipsis_parents = _utils->_last_nodes;
PCFGTryBlock* current_tryblock = _utils->_tryblock_nodes.back();
_utils->_last_nodes = current_tryblock->_handler_parents;
ObjectList<Node*> ellipsis_l = walk(n.get_any());
if(!ellipsis_l.empty())
{
current_tryblock->_nhandlers++;
current_tryblock->_handler_exits.append(_utils->_last_nodes);
for(ObjectList<Node*>::iterator it = current_tryblock->_handler_parents.begin();
it != current_tryblock->_handler_parents.end(); ++it)
{
Edge* catch_edge = (*it)->get_exit_edges().back();
catch_edge->set_catch_edge();
}
}
_utils->_last_nodes = current_tryblock->_handler_exits;
_utils->_tryblock_nodes.pop_back();
if(!try_stmts.empty())
return try_stmts;
else if(!handlers_l.empty())
return handlers_l;
else if(!ellipsis_l.empty())
return ellipsis_l;
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Type& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::Typeid& n)
{
return visit_unary_node(n, n.get_arg());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::UnknownPragma& n)
{
if(VERBOSE)
WARNING_MESSAGE("Ignoring unknown pragma '%s' during PCFG construction",
n.get_text().c_str());
return ObjectList<Node*>();
}
ObjectList<Node*> PCFGVisitor::visit( const Nodecl::VectorAdd& n )
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorAlignRight& n)
{
ObjectList<Node*> all_nodes;
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_left_vector()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_right_vector()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_right_vector()));
_utils->_is_vector = false;
return ObjectList<Node*>(1, merge_nodes(n, all_nodes));
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorArithmeticShr& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorAssignment& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorBitwiseAnd& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorBitwiseNot& n)
{
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorBitwiseOr& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorBitwiseShl& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorBitwiseShr& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorBitwiseXor& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorConditionalExpression& n)
{
_utils->_is_vector = true;
ObjectList<Node*> result = visit_conditional_expression(n);
_utils->_is_vector = false;
return result;
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorConversion& n)
{
_utils->_is_vector = true;
ObjectList<Node*> result = walk(n.get_nest());
_utils->_is_vector = false;
return result;
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorDifferent& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorDiv& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorEqual& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorFabs& n)
{
return visit_vector_unary_node(n, n.get_argument());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorFmadd& n)
{
ObjectList<Node*> all_nodes;
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_first_op()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_second_op()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_third_op()));
_utils->_is_vector = false;
return ObjectList<Node*>(1, merge_nodes(n, all_nodes));
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorFmminus& n)
{
ObjectList<Node*> all_nodes;
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_first_mul_op()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_second_mul_op()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_minus_op()));
_utils->_is_vector = false;
return ObjectList<Node*>(1, merge_nodes(n, all_nodes));
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorFunctionCall& n)
{
return visit_vector_function_call(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorGather& n)
{
return visit_vector_memory_func(n,  '2');
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorGreaterOrEqualThan& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorGreaterThan& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorLiteral& n)
{
_utils->_is_vector = true;
ObjectList<Node*> result = visit_literal_node(n);
_utils->_is_vector = false;
return result;
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorLoad& n)
{
return visit_vector_memory_func(n,  '1');
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorLogicalAnd& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorLogicalNot& n)
{
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorLogicalOr& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorLowerOrEqualThan& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorLowerThan& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskAnd& n)
{   
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskAnd1Not& n)
{   
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskAnd2Not& n)
{   
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskAssignment& n)
{   
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskConversion& n)
{
return walk(n.get_nest());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskNot& n)
{   
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskOr& n)
{   
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMaskXor& n)
{   
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMinus& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMod& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorMul& n)
{
return visit_vector_binary_node(n, n.get_lhs(), n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorNeg& n)
{
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorPrefetch& n)
{
return visit_vector_memory_func(n,  '1');
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorPromotion& n)
{
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorRcp& n)
{
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorReductionAdd& n)
{
Node* reduction = new Node(_utils->_nid, __VectorReduction, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, reduction);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorReductionMinus& n)
{
Node* reduction = new Node(_utils->_nid, __VectorReduction, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, reduction);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorReductionMul& n)
{
Node* reduction = new Node(_utils->_nid, __VectorReduction, _utils->_outer_nodes.top(), n);
return ObjectList<Node*>(1, reduction);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorRsqrt& n)
{
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorScatter& n)
{
return visit_vector_memory_func(n,  '4');
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorSincos& n)
{
ObjectList<Node*> all_nodes;
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_source()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_sin_pointer()));
_utils->_is_vector = true;
all_nodes.insert(walk(n.get_cos_pointer()));
_utils->_is_vector = false;
return ObjectList<Node*>(1, merge_nodes(n, all_nodes));
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorSqrt& n)
{
return visit_vector_unary_node(n, n.get_rhs());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VectorStore& n)
{
return visit_vector_memory_func(n,  '3');
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::VirtualFunctionCall& n)
{
return visit_function_call(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::DefaultArgument& n)
{
return walk(n.get_argument());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranActualArgument& n)
{
return walk(n.get_argument());
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranBozLiteral& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::WhileStatement& n)
{
Node* while_graph_node = _pcfg->create_graph_node(_utils->_outer_nodes.top(), n, __LoopWhile);
_pcfg->connect_nodes(_utils->_last_nodes, while_graph_node);
Node* while_exit = while_graph_node->get_graph_exit_node();
_utils->_last_nodes.clear();
Node* cond_node = walk(n.get_condition())[0];
_pcfg->connect_nodes(while_graph_node->get_graph_entry_node(), cond_node);
_utils->_last_nodes = ObjectList<Node*>(1, cond_node);
while_graph_node->set_condition_node(cond_node);
_utils->_continue_nodes.push(cond_node);
_utils->_break_nodes.push(while_exit);
walk(n.get_statement());    
_utils->_continue_nodes.pop();
_utils->_break_nodes.pop();
int n_conn = _utils->_last_nodes.size();
_pcfg->connect_nodes(_utils->_last_nodes, cond_node,
ObjectList<EdgeType>(n_conn, __Always),
ObjectList<NBase>(n_conn, NBase::null()),
false, true);
ObjectList<Edge*> cond_exits = cond_node->get_exit_edges();
for(ObjectList<Edge*>::iterator it = cond_exits.begin(); it != cond_exits.end(); ++it)
(*it)->set_true_edge();
_pcfg->connect_nodes(cond_node, while_exit, __FalseEdge);
while_exit->set_id(++(_utils->_nid));
while_exit->set_outer_node(_utils->_outer_nodes.top());
_utils->_outer_nodes.pop();
_utils->_last_nodes = ObjectList<Node*>(1, while_graph_node);
return ObjectList<Node*>(1, while_graph_node);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranAllocateStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranDeallocateStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranOpenStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranCloseStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranPrintStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranStopStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranIoStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranWhere& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranReadStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::FortranWriteStatement& n)
{
return visit_literal_node(n);
}
ObjectList<Node*> PCFGVisitor::visit(const Nodecl::CudaKernelCall& n)
{
return visit_literal_node(n);
}
}
}
