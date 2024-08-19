#ifndef TL_EXTENSIBLE_GRAPH_HPP
#define TL_EXTENSIBLE_GRAPH_HPP
#include <algorithm>
#include <map>
#include <stack>
#include "cxx-codegen.h"
#include "cxx-utils.h"
#include "tl-analysis-utils.hpp"
#include "tl-edge.hpp"
#include "tl-node.hpp"
#include "tl-nodecl.hpp"
#include "tl-pcfg-utils.hpp"
namespace TL {
namespace Analysis {
typedef std::map<NBase, NBase, Nodecl::Utils::Nodecl_structural_less> SizeMap;
class LIBTL_CLASS ExtensibleGraph
{
protected:
std::string _name;  
Node* _graph;       
PCFGVisitUtils* _utils;      
const NBase _nodecl;  
Scope _sc;
/
Symbol _function_sym;
Node* _post_sync;
SizeMap _pointer_to_size_map;
std::map<Node*, Node*> nodes_m;
ObjectList<Node*> _task_nodes_l;
ObjectList<Symbol> _func_calls;
std::map<Node*, ObjectList<Node*> > _concurrent_tasks;
std::map<Node*, ObjectList<Node*> > _last_sync_tasks;
std::map<Node*, ObjectList<Node*> > _last_sync_sequential;
std::map<Node*, ObjectList<Node*> > _next_sync_tasks;
std::map<Node*, ObjectList<Node*> > _next_sync_sequential;
std::map<int, int> _cluster_to_entry_map;
bool _usage_computed;
private:
ExtensibleGraph(const ExtensibleGraph& graph);
ExtensibleGraph& operator=(const ExtensibleGraph&);
void clear_unnecessary_nodes();
void remove_unnecessary_connections_rec(Node* n);
void concat_sequential_nodes();
void concat_sequential_nodes_recursive(Node* actual_node, ObjectList<Node*>& last_seq_nodes);
void erase_unclassified_nodes(Node* current);
void erase_jump_nodes(Node* current);
Node* find_nodecl_rec(Node* current, const NBase& n);
Node* find_nodecl_pointer_rec(Node* current, const NBase& n);
void create_and_connect_node(Node* source, Node* target, 
Node* real_source, Node* real_target, 
std::string& dot_graph, std::string& dot_analysis_info,
std::vector<std::vector<std::string> >& outer_edges, 
std::vector<std::vector<Node*> >& outer_nodes, std::string indent);
void get_nodes_dot_data(Node* actual_node, std::string& dot_graph, std::string& dot_analysis_info,
std::vector<std::vector< std::string> >& outer_edges, 
std::vector<std::vector<Node*> >& outer_nodes, std::string indent);
void get_dot_subgraph(Node* actual_node, std::string& graph_data, std::string& graph_analysis_info,
std::vector<std::vector< std::string> >& outer_edges, 
std::vector<std::vector<Node*> >& outer_nodes, std::string indent);
void get_node_dot_data(Node* node, std::string& graph_data, std::string& graph_analysis_info, std::string indent);
void print_node_analysis_info(Node* current, std::string& dot_analysis_info,
std::string cluster_name);
std::string print_pragma_node_clauses(Node* current, std::string indent, std::string cluster_name);
public:
ExtensibleGraph(std::string name, const NBase& nodecl, PCFGVisitUtils* utils);
Node* append_new_child_to_parent(ObjectList<Node*> parents, ObjectList<NBase> stmts,
NodeType ntype = __Normal, EdgeType etype = __Always);
Node* append_new_child_to_parent(Node* parent, NBase stmt,
NodeType ntype = __Normal, EdgeType etype = __Always);
Node* append_new_child_to_parent(ObjectList<Node*> parents, NBase stmt,
NodeType ntype = __Normal, EdgeType etype = __Always);
Edge* connect_nodes(Node* parent, Node* child, EdgeType etype = __Always,
const NBase& label = NBase::null(),
bool is_task_edge = false, bool is_back_edge = false);
void connect_nodes(const ObjectList<Node*>& parents, const ObjectList<Node*>& children,
const ObjectList<EdgeType>& etypes=ObjectList<EdgeType>(),
const ObjectList<NBase>& elabels=ObjectList<NBase>());
void connect_nodes(Node* parent, const ObjectList<Node*>& children,
const ObjectList<EdgeType>& etypes=ObjectList<EdgeType>(),
const ObjectList<NBase>& elabels=ObjectList<NBase>());
void connect_nodes(const ObjectList<Node*>& parents, Node* child, 
const ObjectList<EdgeType>& etypes=ObjectList<EdgeType>(),
const ObjectList<NBase>& elabels=ObjectList<NBase>(), 
bool is_task_edge = false, bool is_back_edge = false);
void disconnect_nodes(ObjectList<Node*> parents, Node* child);
void disconnect_nodes(Node* parent, ObjectList<Node*> children);
void disconnect_nodes(Node *parent, Node *child);
Node* create_graph_node(Node* outer_node, NBase label,
GraphType graph_type, NBase context = NBase::null());
Node* create_flush_node(Node* outer_node, NBase n = NBase::null());
Node* create_unconnected_node(NodeType type, NBase nodecl);
void delete_node(Node* n);
void dress_up_graph();
void concat_nodes(ObjectList<Node*> node_l);
static void clear_visits(Node* node);
static void clear_visits_aux(Node* node);
static void clear_visits_extgraph(Node* node);
static void clear_visits_extgraph_aux(Node* node);
static void clear_visits_in_level(Node* node, Node* outer_node);
static void clear_visits_in_level_no_nest(Node* node, Node* outer_node);
static void clear_visits_aux_in_level(Node* node, Node* outer_node);
static void clear_visits_backwards_in_level(Node* node, Node* graph);
static void clear_visits_backwards(Node* node);
void print_graph_to_dot(bool usage = false, bool liveness = false, 
bool reaching_defs = false, bool induction_vars = false, bool ranges = false,
bool auto_scoping = false, bool auto_deps = false);
std::string get_name() const;
NBase get_nodecl() const;
Scope get_scope() const;
const NodeclSet& get_global_variables() const;
void set_global_vars(const NodeclSet& global_vars);
Symbol get_function_symbol() const;
Node* get_post_sync() const;
void set_post_sync(Node* post_sync);
void set_pointer_n_elems(const NBase& s, const NBase& size);
NBase get_pointer_n_elems(const NBase& s);
SizeMap get_pointer_n_elements_map();
void purge_non_constant_pointer_n_elems();
Node* get_graph() const;
ObjectList<Node*> get_tasks_list() const;
ObjectList<Symbol> get_function_parameters() const;
void add_func_call_symbol(Symbol s);
ObjectList<Symbol> get_function_calls() const;
ObjectList<Node*> get_task_concurrent_tasks(Node* task) const;
void add_concurrent_task_group(Node* task, ObjectList<Node*> concurrent_tasks);
ObjectList<Node*> get_task_last_sync_for_tasks(Node* task) const;
ObjectList<Node*> get_task_last_sync_for_sequential_code(Node* task) const;
void add_last_sync_for_tasks(Node* task, Node* last_sync);
void set_last_sync_for_tasks(Node* task, Node* last_sync);
void add_last_sync_for_sequential_code(Node* task, Node* last_sync);
ObjectList<Node*> get_task_next_sync_for_tasks(Node* task) const;
ObjectList<Node*> get_task_next_sync_for_sequential_code(Node* task) const;
void add_next_sync_for_tasks(Node* task, Node* next_sync);
void add_next_sync_for_sequential_code(Node* task, Node* next_sync);
void remove_next_sync_for_tasks(Node* task, Node* next_sync);
void remove_concurrent_task(Node* task, Node* old_concurrent_task);
static Node* is_for_loop_increment(Node* node);
static bool node_is_in_loop(Node* current);
static bool node_is_in_conditional_branch(Node* current, Node* max_outer = NULL);
static bool node_is_in_synchronous_construct(Node* current);
static bool is_backward_parent(Node* son, Node* parent);
static bool node_contains_node(Node* container, Node* contained);
static Node* get_extensible_graph_from_node(Node* node);
static bool node_is_ancestor_of_node(Node* ancestor, Node* descendant);
static Node* get_omp_enclosing_node(Node* current);
static Edge* get_edge_between_nodes(Node* source, Node* target);
static Node* get_enclosing_context(Node* n);
static Node* get_most_outer_parallel(Node* n);
static Node* get_most_outer_loop(Node* n);
static Node* get_enclosing_task(Node* n);
static bool task_encloses_task(Node* container, Node* contained);
static bool node_contains_tasks(Node* graph_node, Node* current, ObjectList<Node*>& tasks);
static bool node_contains_task_sync(Node* graph_node, Node* current);
static Node* get_enclosing_control_structure(Node* node);
static Node* get_task_creation_from_task(Node* task);
static Node* get_task_from_task_creation(Node* task_creation);
static bool task_synchronizes_in_post_sync(Node* task);
static bool has_been_defined(Node* current, Node* scope, const NBase& n);
Node* find_nodecl(const NBase& n);           
Node* find_nodecl_pointer(const NBase& n);   
bool usage_is_computed() const;
void set_usage_computed();
friend class PCFGVisitor;
};
}
}
#endif 
