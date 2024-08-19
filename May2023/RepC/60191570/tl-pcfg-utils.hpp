#ifndef TL_PCFGVISIT_UTILS_HPP
#define TL_PCFGVISIT_UTILS_HPP
#include "tl-analysis-utils.hpp"
#include "tl-nodecl-utils.hpp"
#include "tl-objectlist.hpp"
#include <set>
#include <stack>
namespace TL {
namespace Analysis {
#define NODE_TYPE_LIST \
NODE_TYPE(UnclassifiedNode) \
NODE_TYPE(AsmOp) \
NODE_TYPE(Break) \
NODE_TYPE(Builtin) \
NODE_TYPE(Continue) \
NODE_TYPE(Entry) \
NODE_TYPE(Exit) \
NODE_TYPE(FunctionCall) \
NODE_TYPE(Goto) \
NODE_TYPE(Labeled) \
NODE_TYPE(Normal) \
NODE_TYPE(OmpBarrier) \
NODE_TYPE(OmpFlush) \
NODE_TYPE(OmpTaskCreation) \
NODE_TYPE(OmpTaskwait) \
NODE_TYPE(OmpTaskyield) \
NODE_TYPE(OmpVirtualTaskSync) \
NODE_TYPE(OmpWaitonDeps) \
NODE_TYPE(Return) \
NODE_TYPE(VectorFunctionCall) \
NODE_TYPE(VectorGather) \
NODE_TYPE(VectorLoad) \
NODE_TYPE(VectorNormal) \
NODE_TYPE(VectorReduction) \
NODE_TYPE(VectorScatter) \
NODE_TYPE(VectorStore) \
NODE_TYPE(Graph)
enum NodeType {
#undef NODE_TYPE
#define NODE_TYPE(X) __##X,
NODE_TYPE_LIST
#undef NODE_TYPE
};
#define GRAPH_NODE_TYPE_LIST \
GRAPH_TYPE(AsmDef) \
GRAPH_TYPE(CondExpr) \
GRAPH_TYPE(Context) \
GRAPH_TYPE(ExtensibleGraph) \
GRAPH_TYPE(FunctionCallGraph) \
GRAPH_TYPE(FunctionCode) \
GRAPH_TYPE(IfElse) \
GRAPH_TYPE(LoopDoWhile) \
GRAPH_TYPE(LoopFor) \
GRAPH_TYPE(LoopWhile) \
GRAPH_TYPE(OmpAsyncTarget) \
GRAPH_TYPE(OmpAtomic) \
GRAPH_TYPE(OmpBarrierGraph) \
GRAPH_TYPE(OmpCritical) \
GRAPH_TYPE(OmpForAppendix) \
GRAPH_TYPE(OmpLoop) \
GRAPH_TYPE(OmpMaster) \
GRAPH_TYPE(OmpParallel) \
GRAPH_TYPE(OmpSection) \
GRAPH_TYPE(OmpSections) \
GRAPH_TYPE(OmpSimd) \
GRAPH_TYPE(OmpSimdFor) \
GRAPH_TYPE(OmpSimdFunction) \
GRAPH_TYPE(OmpSimdParallelFor) \
GRAPH_TYPE(OmpSingle) \
GRAPH_TYPE(OmpssLint) \
GRAPH_TYPE(OmpSyncTarget) \
GRAPH_TYPE(OmpWorkshare) \
GRAPH_TYPE(OmpTask) \
GRAPH_TYPE(SplitStmt) \
GRAPH_TYPE(Switch) \
GRAPH_TYPE(SwitchCase) \
GRAPH_TYPE(VectorCondExpr) \
GRAPH_TYPE(VectorFunctionCallGraph)
enum GraphType {
#undef GRAPH_TYPE
#define GRAPH_TYPE(X) __##X,
GRAPH_NODE_TYPE_LIST
#undef GRAPH_TYPE
};
#define EDGE_TYPE_LIST \
EDGE_TYPE(UnclassifiedEdge) \
EDGE_TYPE(Always) \
EDGE_TYPE(Case) \
EDGE_TYPE(Catch) \
EDGE_TYPE(FalseEdge) \
EDGE_TYPE(GotoEdge) \
EDGE_TYPE(TrueEdge)
enum EdgeType {
#undef EDGE_TYPE
#define EDGE_TYPE(X) __##X,
EDGE_TYPE_LIST
#undef EDGE_TYPE
};
#define SYNC_KIND_LIST \
SYNC_KIND(Static) \
SYNC_KIND(Maybe) \
SYNC_KIND(Post)
enum SyncKind {
#undef SYNC_KIND
#define SYNC_KIND(X) __##X,
SYNC_KIND_LIST
#undef SYNC_KIND
};
enum ASM_node_info {
ASM_DEF_TEXT,
ASM_DEF_INPUT_OPS,
ASM_DEF_OUTPUT_OPS,
ASM_DEF_CLOBBERED_REGS,
ASM_OP_CONSTRAINT,
ASM_OP_EXPRESSION
};
enum PCFGAttribute {
_NODE_LABEL,
_NODE_STMTS,
_ENTRY_NODE,
_EXIT_NODE,
_CONDITION_NODE,
_STRIDE_NODE,
_GRAPH_TYPE,
_OMP_INFO,
_ASM_INFO,
_TASK_CONTEXT,
_TASK_FUNCTION,
_CLAUSES,
_ARGS,
_EDGE_TYPE,
_EDGE_LABEL,
_SYNC_KIND,
_CONDITION,
_IS_TASK_EDGE,
_IS_BACK_EDGE,
_LATTICE_VALS,
_IS_EXECUTABLE,
_IS_EXECUTABLE_EDGE,
_LIVE_IN_TASKS,
_LIVE_OUT_TASKS,
_UPPER_EXPOSED,
_KILLED,
_UNDEF,
_PRIVATE_UPPER_EXPOSED,
_PRIVATE_KILLED,
_PRIVATE_UNDEF,
_USED_ADDRESSES,
_LIVE_IN,
_LIVE_OUT,
_GEN,
_REACH_DEFS_IN,
_REACH_DEFS_OUT,
_AUX_REACH_DEFS,
_INDUCTION_VARS,
_RANGES,
_SC_AUTO,
_SC_SHARED,
_SC_PRIVATE,
_SC_FIRSTPRIVATE,
_SC_UNDEF,
_SC_RACE,
_DEPS_PRIVATE,
_DEPS_FIRSTPRIVATE,
_DEPS_SHARED,
_DEPS_IN,
_DEPS_OUT,
_DEPS_INOUT,
_DEPS_UNDEF,
_CORRECTNESS_AUTO_STORAGE_VARS,
_CORRECTNESS_DEAD_VARS,
_CORRECTNESS_INCOHERENT_FP_VARS,
_CORRECTNESS_INCOHERENT_IN_VARS,
_CORRECTNESS_INCOHERENT_IN_POINTED_VARS,
_CORRECTNESS_INCOHERENT_OUT_VARS,
_CORRECTNESS_INCOHERENT_OUT_POINTED_VARS,
_CORRECTNESS_INCOHERENT_P_VARS,
_CORRECTNESS_RACE_VARS,
_CORRECTNESS_UNNECESSARILY_SCOPED_VARS,
_ASSERT_UPPER_EXPOSED,
_ASSERT_KILLED,
_ASSERT_UNDEFINED,
_ASSERT_LIVE_IN,
_ASSERT_LIVE_OUT,
_ASSERT_DEAD,
_ASSERT_REACH_DEFS_IN,
_ASSERT_REACH_DEFS_OUT,
_ASSERT_INDUCTION_VARS,
_ASSERT_AUTOSC_FIRSTPRIVATE,
_ASSERT_AUTOSC_PRIVATE,
_ASSERT_AUTOSC_SHARED,
_ASSERT_RANGE,
_ASSERT_CORRECTNESS_AUTO_STORAGE_VARS,
_ASSERT_CORRECTNESS_INCOHERENT_FP_VARS,
_ASSERT_CORRECTNESS_INCOHERENT_IN_VARS,
_ASSERT_CORRECTNESS_INCOHERENT_IN_POINTED_VARS,
_ASSERT_CORRECTNESS_INCOHERENT_OUT_VARS,
_ASSERT_CORRECTNESS_INCOHERENT_OUT_POINTED_VARS,
_ASSERT_CORRECTNESS_INCOHERENT_P_VARS,
_ASSERT_CORRECTNESS_RACE_VARS,
_ASSERT_CORRECTNESS_DEAD_VARS
};
class Node;
class PCFGLoopControl
{
private:
Node* _init;
Node* _cond;
Node* _next;
public:
PCFGLoopControl();
~PCFGLoopControl();
friend class PCFGVisitor;
};
class PCFGTryBlock
{
private:
ObjectList<Node*> _handler_parents;
ObjectList<Node*> _handler_exits;
int _nhandlers;
public:
PCFGTryBlock();
~PCFGTryBlock();
friend class PCFGVisitor;
};
class PCFGSwitch
{
private:
Node* _condition;       
Node* _exit;
public:
PCFGSwitch(Node* condition, Node* exit);
~PCFGSwitch();
void set_condition(Node* condition);
friend class PCFGVisitor;
};
class PCFGPragmaInfo
{
private:
ObjectList<NBase> _clauses;
public:
PCFGPragmaInfo(const NBase& clause);
PCFGPragmaInfo();
PCFGPragmaInfo(const PCFGPragmaInfo& p);
bool has_clause(node_t kind) const;
NBase get_clause(node_t kind) const;
void add_clause(const NBase& clause);
ObjectList<NBase> get_clauses() const;
friend class PCFGVisitor;
};
class LIBTL_CLASS PCFGVisitUtils
{
private:
ObjectList<Node*> _last_nodes;
ObjectList<Node*> _return_nodes;
std::stack<Node*> _outer_nodes;
std::stack<Node*> _continue_nodes;
std::stack<Node*> _break_nodes;
ObjectList<Node*> _labeled_nodes;
ObjectList<Node*> _goto_nodes;
std::stack<PCFGSwitch*> _switch_nodes;
std::stack<PCFGLoopControl*> _nested_loop_nodes;
ObjectList<PCFGTryBlock*> _tryblock_nodes;
std::stack<PCFGPragmaInfo> _pragma_nodes;
std::stack<NBase> _context_nodecl;
std::stack<ObjectList<Node*> > _section_nodes;
std::stack<Node*> _assert_nodes;
std::stack<std::pair<Node*, Node*> > _environ_entry_exit;
bool _is_vector;
unsigned int _nid;
public:
PCFGVisitUtils();
friend class ExtensibleGraph;
friend class PCFGVisitor;
};
std::string print_node_list(const ObjectList<Node*>& list);
struct AliveTaskItem
{
Node* node;
int domain;
AliveTaskItem(Node* node_, int domain_)
: node(node_), domain(domain_)
{}
bool operator<(const AliveTaskItem& it) const
{
return (this->node < it.node)
|| (!(it.node < this->node) && 
(this->domain < it.domain));
}
bool operator==(const AliveTaskItem& it) const
{
return (this->node == it.node)
&& (this->domain == it.domain);
}
};
typedef std::set<AliveTaskItem> AliveTaskSet;
class Node;
class Edge;
typedef ObjectList<Node*> NodeList;
typedef ObjectList<Edge*> EdgeList;
typedef ObjectList<EdgeType> EdgeTypeList;
}
}
#endif          
