

#define HARNESS_DEFAULT_MIN_THREADS 3
#define HARNESS_DEFAULT_MAX_THREADS 4

#if _MSC_VER
#pragma warning (disable: 4503) 
#if _MSC_VER==1700 && !defined(__INTEL_COMPILER)
#pragma warning (disable: 4702)
#endif
#endif

#define TBB_DEPRECATED_INPUT_NODE_BODY __TBB_CPF_BUILD

#include "harness.h"
#include <string> 

#include "tbb/spin_mutex.h"
#include "tbb/spin_rw_mutex.h"
#include "tbb/task.h"
#include "tbb/task_arena.h"

#define private public
#define protected public
#include "tbb/flow_graph.h"
#undef protected
#undef private
#include "tbb/task_scheduler_init.h"
#include "harness_graph.h"

template<typename T>
struct receiverBody {
tbb::flow::continue_msg operator()(const T &) {
return tbb::flow::continue_msg();
}
};

void TestSplitNode() {
typedef tbb::flow::split_node<tbb::flow::tuple<int> > snode_type;
tbb::flow::graph g;
snode_type snode(g);
tbb::flow::function_node<int> rcvr(g,tbb::flow::unlimited, receiverBody<int>());
REMARK("Testing split_node\n");
ASSERT(tbb::flow::output_port<0>(snode).my_successors.empty(), "Constructed split_node has successors");
tbb::flow::make_edge(tbb::flow::output_port<0>(snode), rcvr);
ASSERT(!(tbb::flow::output_port<0>(snode).my_successors.empty()), "after make_edge, split_node has no successor.");
snode.try_put(tbb::flow::tuple<int>(1));
g.wait_for_all();
g.reset();
ASSERT(!(tbb::flow::output_port<0>(snode).my_successors.empty()), "after reset(), split_node has no successor.");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(tbb::flow::output_port<0>(snode).my_successors.empty(), "after reset(rf_clear_edges), split_node has a successor.");
}

template< typename B >
void TestBufferingNode(const char * name) {
tbb::flow::graph g;
B                bnode(g);
tbb::flow::function_node<int,int,tbb::flow::rejecting> fnode(g, tbb::flow::serial, serial_fn_body<int>(serial_fn_state0));
REMARK("Testing %s:", name);
for(int icnt = 0; icnt < 2; icnt++) {
bool reverse_edge = (icnt & 0x2) != 0;
serial_fn_state0 = 0;  
REMARK(" make_edge");
tbb::flow::make_edge(bnode, fnode);
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after make_edge");
REMARK(" try_put");
bnode.try_put(1);  
BACKOFF_WAIT(serial_fn_state0 == 0, "Timed out waiting for first put");
if(reverse_edge) {
REMARK(" try_put2");
bnode.try_put(2);  
BACKOFF_WAIT(!bnode.my_successors.empty(), "Timed out waiting after 2nd put");
ASSERT(bnode.my_successors.empty(), "successor not removed");
}
else {
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after forwarding message");
}
serial_fn_state0 = 0;  
if(reverse_edge) {
BACKOFF_WAIT( serial_fn_state0 == 0, "Timed out waiting after 2nd put");
serial_fn_state0 = 0;  
}
g.wait_for_all();
REMARK(" remove_edge");
tbb::flow::remove_edge(bnode, fnode);
ASSERT(bnode.my_successors.empty(), "buffering node has a successor after remove_edge");
}
tbb::flow::join_node<tbb::flow::tuple<int,int>,tbb::flow::reserving> jnode(g);
tbb::flow::make_edge(bnode, tbb::flow::input_port<0>(jnode));  
g.wait_for_all();
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after attaching to join");
REMARK(" reverse");
bnode.try_put(1);  
g.wait_for_all();
ASSERT(bnode.my_successors.empty(), "buffering node has a successor after reserving");
REMARK(" reset()");
g.wait_for_all();
g.reset();  
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after reset()");
REMARK(" remove_edge");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(bnode.my_successors.empty(), "buffering node has a successor after reset(rf_clear_edges)");
tbb::flow::make_edge(bnode, tbb::flow::input_port<0>(jnode));  
bnode.try_put(1);  
g.wait_for_all();
ASSERT(bnode.my_successors.empty(), "buffering node has a successor after reserving");
REMARK(" remove_edge(reversed)");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(bnode.my_successors.empty(), "buffering node has no successor after reset()");
ASSERT(tbb::flow::input_port<0>(jnode).my_predecessors.empty(), "predecessor not reset");
REMARK("  done\n");
g.wait_for_all();
}

void TestContinueNode() {
tbb::flow::graph g;
tbb::flow::function_node<int> fnode0(g, tbb::flow::serial, serial_fn_body<int>(serial_fn_state0));
tbb::flow::continue_node<int> cnode(g, 1, serial_continue_body<int>(serial_continue_state0));
tbb::flow::function_node<int> fnode1(g, tbb::flow::serial, serial_fn_body<int>(serial_fn_state1));
tbb::flow::make_edge(fnode0, cnode);
tbb::flow::make_edge(cnode, fnode1);
REMARK("Testing continue_node:");
for( int icnt = 0; icnt < 2; ++icnt ) {
REMARK( " initial%d", icnt);
ASSERT(cnode.my_predecessor_count == 2, "predecessor addition didn't increment count");
ASSERT(!cnode.successors().empty(), "successors empty though we added one");
ASSERT(cnode.my_current_count == 0, "state of continue_receiver incorrect");
serial_continue_state0 = 0;
serial_fn_state0 = 0;
serial_fn_state1 = 0;

fnode0.try_put(1);  
BACKOFF_WAIT(!serial_fn_state0, "Timed out waiting for function_node to start");
serial_fn_state0 = 0;  
BACKOFF_WAIT(serial_continue_state0 == 0 && cnode.my_current_count == 0, "Timed out waiting for continue_state0 to change");
ASSERT(serial_continue_state0 == 0, "Improperly released continue_node");
ASSERT(cnode.my_current_count == 1, "state of continue_receiver incorrect");
if(icnt == 0) {  
REMARK(" firing");
fnode0.try_put(1);  
BACKOFF_WAIT(serial_fn_state0 == 0, "timeout waiting for continue_body to execute");
serial_fn_state0 = 0;  

BACKOFF_WAIT(!serial_continue_state0,"continue_node didn't start");  
ASSERT(cnode.my_current_count == 0, " my_current_count not reset before body of continue_node started");
serial_continue_state0 = 0;  
BACKOFF_WAIT(!serial_fn_state1,"successor function_node didn't start");    
serial_fn_state1 = 0;  
g.wait_for_all();

{
int i;
ASSERT(!cnode.try_get(i), "try_get not rejected");
}

REMARK(" reset");
ASSERT(!cnode.my_successors.empty(), "Empty successors in built graph (before reset)");
ASSERT(cnode.my_predecessor_count == 2, "predecessor_count reset (before reset)");
g.reset();  
ASSERT(!cnode.my_successors.empty(), "Empty successors in built graph (after reset)" );
ASSERT(cnode.my_predecessor_count == 2, "predecessor_count reset (after reset)");
}
else {  
g.wait_for_all();
REMARK(" reset(rf_clear_edges)");
ASSERT(!cnode.my_successors.empty(), "Empty successors in built graph (before reset)");
ASSERT(cnode.my_predecessor_count == 2, "predecessor_count reset (before reset)");
g.reset(tbb::flow::rf_clear_edges);  
ASSERT(cnode.my_current_count == 0, "state of continue_receiver incorrect after reset(rf_clear_edges)");
ASSERT(cnode.my_successors.empty(), "buffering node has a successor after reset(rf_clear_edges)");
ASSERT(cnode.my_predecessor_count == cnode.my_initial_predecessor_count, "predecessor count not reset");
}
}

REMARK(" done\n");

}

void TestFunctionNode() {
tbb::flow::graph g;
tbb::flow::queue_node<int> qnode0(g);
tbb::flow::function_node<int,int, tbb::flow::rejecting > fnode0(g, tbb::flow::serial, serial_fn_body<int>(serial_fn_state0));
tbb::flow::function_node<int,int> fnode1(g, tbb::flow::serial, serial_fn_body<int>(serial_fn_state0));

tbb::flow::queue_node<int> qnode1(g);

tbb::flow::make_edge(fnode0, qnode1);
tbb::flow::make_edge(qnode0, fnode0);

serial_fn_state0 = 2;  
qnode0.try_put(1);
g.wait_for_all();
int ii;
ASSERT(qnode1.try_get(ii) && ii == 1, "output not passed");
tbb::flow::remove_edge(qnode0, fnode0);
tbb::flow::remove_edge(fnode0, qnode1);

tbb::flow::make_edge(fnode1, qnode1);
tbb::flow::make_edge(qnode0, fnode1);

serial_fn_state0 = 2;  
qnode0.try_put(1);
g.wait_for_all();
ASSERT(qnode1.try_get(ii) && ii == 1, "output not passed");
tbb::flow::remove_edge(qnode0, fnode1);
tbb::flow::remove_edge(fnode1, qnode1);

serial_fn_state0 = 0;
tbb::flow::make_edge(fnode0, qnode1);
tbb::flow::make_edge(qnode0, fnode0);
REMARK("Testing rejecting function_node:");
ASSERT(!fnode0.my_queue, "node should have no queue");
ASSERT(!fnode0.my_successors.empty(), "successor edge not added");
qnode0.try_put(1);
BACKOFF_WAIT(!serial_fn_state0,"rejecting function_node didn't start");
qnode0.try_put(2);   
BACKOFF_WAIT(fnode0.my_predecessors.empty(), "Missing predecessor ---");
serial_fn_state0 = 2;   
g.wait_for_all();
REMARK(" reset");
g.reset();  
ASSERT(!qnode0.my_successors.empty(), "empty successors after reset()");
ASSERT(fnode0.my_predecessors.empty(), "predecessor not reversed");
tbb::flow::remove_edge(qnode0, fnode0);
tbb::flow::remove_edge(fnode0, qnode1);
REMARK("\n");

tbb::flow::make_edge(fnode1, qnode1);
REMARK("Testing queueing function_node:");
ASSERT(fnode1.my_queue, "node should have no queue");
ASSERT(!fnode1.my_successors.empty(), "successor edge not added");
REMARK(" add_pred");
ASSERT(fnode1.register_predecessor(qnode0), "Cannot register as predecessor");
ASSERT(!fnode1.my_predecessors.empty(), "Missing predecessor");
REMARK(" reset");
g.wait_for_all();
g.reset();  
ASSERT(!qnode0.my_successors.empty(), "empty successors after reset()");
ASSERT(fnode1.my_predecessors.empty(), "predecessor not reversed");
tbb::flow::remove_edge(qnode0, fnode1);
tbb::flow::remove_edge(fnode1, qnode1);
REMARK("\n");

serial_fn_state0 = 0;  
tbb::flow::make_edge(qnode0, fnode0);
REMARK(" start_func");
qnode0.try_put(1);
BACKOFF_WAIT(serial_fn_state0 == 0, "Timed out waiting after 1st put");
REMARK(" put_node(2)");
qnode0.try_put(2);   
BACKOFF_WAIT(fnode0.my_predecessors.empty(), "Timed out waiting");
ASSERT(!fnode0.my_predecessors.empty(), "function_node edge not reversed");
g.my_root_task->cancel_group_execution();
serial_fn_state0 = 2;
g.wait_for_all();
ASSERT(!fnode0.my_predecessors.empty() && qnode0.my_successors.empty(), "function_node edge not reversed");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(fnode0.my_predecessors.empty() && qnode0.my_successors.empty(), "function_node edge not removed");
ASSERT(fnode0.my_successors.empty(), "successor to fnode not removed");
REMARK(" done\n");
}

template<typename TT>
class tag_func {
TT my_mult;
public:
tag_func(TT multiplier) : my_mult(multiplier) { }
tbb::flow::tag_value operator()( TT v) {
tbb::flow::tag_value t = tbb::flow::tag_value(v / my_mult);
return t;
}
};

template<typename JNODE_TYPE>
void
TestSimpleSuccessorArc(const char *name) {
tbb::flow::graph g;
{
REMARK("Join<%s> successor test ", name);
tbb::flow::join_node<tbb::flow::tuple<int>, JNODE_TYPE> qj(g);
tbb::flow::broadcast_node<tbb::flow::tuple<int> > bnode(g);
tbb::flow::make_edge(qj, bnode);
ASSERT(!qj.my_successors.empty(),"successor missing after linking");
g.reset();
ASSERT(!qj.my_successors.empty(),"successor missing after reset()");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(qj.my_successors.empty(), "successors not removed after reset(rf_clear_edges)");
}
}

template<>
void
TestSimpleSuccessorArc<tbb::flow::tag_matching>(const char *name) {
tbb::flow::graph g;
{
REMARK("Join<%s> successor test ", name);
typedef tbb::flow::tuple<int,int> my_tuple;
tbb::flow::join_node<my_tuple, tbb::flow::tag_matching> qj(g,
tag_func<int>(1),
tag_func<int>(1)
);
tbb::flow::broadcast_node<my_tuple > bnode(g);
tbb::flow::make_edge(qj, bnode);
ASSERT(!qj.my_successors.empty(),"successor missing after linking");
g.reset();
ASSERT(!qj.my_successors.empty(),"successor missing after reset()");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(qj.my_successors.empty(), "successors not removed after reset(rf_clear_edges)");
}
}

void
TestJoinNode() {
tbb::flow::graph g;

TestSimpleSuccessorArc<tbb::flow::queueing>("queueing");
TestSimpleSuccessorArc<tbb::flow::reserving>("reserving");
TestSimpleSuccessorArc<tbb::flow::tag_matching>("tag_matching");

REMARK(" reserving preds");
{
tbb::flow::join_node<tbb::flow::tuple<int,int>, tbb::flow::reserving> rj(g);
tbb::flow::queue_node<int> q0(g);
tbb::flow::queue_node<int> q1(g);
tbb::flow::make_edge(q0,tbb::flow::input_port<0>(rj));
tbb::flow::make_edge(q1,tbb::flow::input_port<1>(rj));
q0.try_put(1);
g.wait_for_all();  
ASSERT(!(tbb::flow::input_port<0>(rj).my_predecessors.empty()),"reversed port missing predecessor");
ASSERT((tbb::flow::input_port<1>(rj).my_predecessors.empty()),"non-reversed port has pred");
g.reset();
ASSERT((tbb::flow::input_port<0>(rj).my_predecessors.empty()),"reversed port has pred after reset()");
ASSERT((tbb::flow::input_port<1>(rj).my_predecessors.empty()),"non-reversed port has pred after reset()");
q1.try_put(2);
g.wait_for_all();  
ASSERT(!(tbb::flow::input_port<1>(rj).my_predecessors.empty()),"reversed port missing predecessor");
ASSERT((tbb::flow::input_port<0>(rj).my_predecessors.empty()),"non-reversed port has pred");
g.reset();
ASSERT((tbb::flow::input_port<1>(rj).my_predecessors.empty()),"reversed port has pred after reset()");
ASSERT((tbb::flow::input_port<0>(rj).my_predecessors.empty()),"non-reversed port has pred after reset()");
q1.try_put(3);
g.wait_for_all();  
ASSERT(!(tbb::flow::input_port<1>(rj).my_predecessors.empty()),"reversed port missing predecessor");
ASSERT((tbb::flow::input_port<0>(rj).my_predecessors.empty()),"non-reversed port has pred");
g.reset(tbb::flow::rf_clear_edges);
ASSERT((tbb::flow::input_port<1>(rj).my_predecessors.empty()),"reversed port has pred after reset()");
ASSERT((tbb::flow::input_port<0>(rj).my_predecessors.empty()),"non-reversed port has pred after reset()");
ASSERT(q0.my_successors.empty(), "edge not removed by reset(rf_clear_edges)");
ASSERT(q1.my_successors.empty(), "edge not removed by reset(rf_clear_edges)");
}
REMARK(" done\n");
}

void
TestLimiterNode() {
int out_int;
tbb::flow::graph g;
tbb::flow::limiter_node<int> ln(g,1);
REMARK("Testing limiter_node: preds and succs");
ASSERT(ln.decrement.my_predecessor_count == 0, "error in pred count");
ASSERT(ln.decrement.my_initial_predecessor_count == 0, "error in initial pred count");
ASSERT(ln.decrement.my_current_count == 0, "error in current count");
#if TBB_DEPRECATED_LIMITER_NODE_CONSTRUCTOR
ASSERT(ln.init_decrement_predecessors == 0, "error in decrement predecessors");
#endif
ASSERT(ln.my_threshold == 1, "error in my_threshold");
tbb::flow::queue_node<int> inq(g);
tbb::flow::queue_node<int> outq(g);
tbb::flow::broadcast_node<tbb::flow::continue_msg> bn(g);

tbb::flow::make_edge(inq,ln);
tbb::flow::make_edge(ln,outq);
tbb::flow::make_edge(bn,ln.decrement);

g.wait_for_all();
ASSERT(!(ln.my_successors.empty()),"successors empty after make_edge");
ASSERT(ln.my_predecessors.empty(), "input edge reversed");
inq.try_put(1);
g.wait_for_all();
ASSERT(outq.try_get(out_int) && out_int == 1, "limiter_node didn't pass first value");
ASSERT(ln.my_predecessors.empty(), "input edge reversed");
inq.try_put(2);
g.wait_for_all();
ASSERT(!outq.try_get(out_int), "limiter_node incorrectly passed second input");
ASSERT(!ln.my_predecessors.empty(), "input edge to limiter_node not reversed");
bn.try_put(tbb::flow::continue_msg());
g.wait_for_all();
ASSERT(outq.try_get(out_int) && out_int == 2, "limiter_node didn't pass second value");
g.wait_for_all();
ASSERT(!ln.my_predecessors.empty(), "input edge was reversed(after try_get())");
g.reset();
ASSERT(ln.my_predecessors.empty(), "input edge not reset");
inq.try_put(3);
g.wait_for_all();
ASSERT(outq.try_get(out_int) && out_int == 3, "limiter_node didn't pass third value");

REMARK(" rf_clear_edges");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(ln.decrement.my_predecessor_count == 0, "error in pred count");
ASSERT(ln.decrement.my_initial_predecessor_count == 0, "error in initial pred count");
ASSERT(ln.decrement.my_current_count == 0, "error in current count");
#if TBB_DEPRECATED_LIMITER_NODE_CONSTRUCTOR
ASSERT(ln.init_decrement_predecessors == 0, "error in decrement predecessors");
#endif
ASSERT(ln.my_threshold == 1, "error in my_threshold");
ASSERT(ln.my_predecessors.empty(), "preds not reset(rf_clear_edges)");
ASSERT(ln.my_successors.empty(), "preds not reset(rf_clear_edges)");
ASSERT(inq.my_successors.empty(), "Arc not removed on reset(rf_clear_edges)");
ASSERT(inq.my_successors.empty(), "Arc not removed on reset(rf_clear_edges)");
ASSERT(bn.my_successors.empty(), "control edge not removed on reset(rf_clear_edges)");
tbb::flow::make_edge(inq,ln);
tbb::flow::make_edge(ln,outq);
inq.try_put(4);
inq.try_put(5);
g.wait_for_all();
ASSERT(outq.try_get(out_int),"missing output after reset(rf_clear_edges)");
ASSERT(out_int == 4, "input incorrect (4)");
bn.try_put(tbb::flow::continue_msg());
g.wait_for_all();
ASSERT(!outq.try_get(out_int),"second output incorrectly passed (rf_clear_edges)");
REMARK(" done\n");
}

template<typename MF_TYPE>
struct mf_body {
tbb::atomic<int> *_flag;
mf_body( tbb::atomic<int> &myatomic) : _flag(&myatomic) { }
void operator()( const int& in, typename MF_TYPE::output_ports_type &outports) {
if(*_flag == 0) {
*_flag = 1;
BACKOFF_WAIT(*_flag == 1, "multifunction_node not released");
}

if(in & 0x1) tbb::flow::get<1>(outports).try_put(in);
else         tbb::flow::get<0>(outports).try_put(in);
}
};

template<typename P, typename T>
struct test_reversal;
template<typename T>
struct test_reversal<tbb::flow::queueing, T> {
test_reversal() { REMARK("<queueing>"); }
bool operator()( T &node) { return node.my_predecessors.empty(); }
};

template<typename T>
struct test_reversal<tbb::flow::rejecting, T> {
test_reversal() { REMARK("<rejecting>"); }
bool operator()( T &node) { return !node.my_predecessors.empty(); }
};

template<typename P>
void
TestMultifunctionNode() {
typedef tbb::flow::multifunction_node<int, tbb::flow::tuple<int, int>, P> multinode_type;
REMARK("Testing multifunction_node");
test_reversal<P,multinode_type> my_test;
REMARK(":");
tbb::flow::graph g;
multinode_type mf(g, tbb::flow::serial, mf_body<multinode_type>(serial_fn_state0));
tbb::flow::queue_node<int> qin(g);
tbb::flow::queue_node<int> qodd_out(g);
tbb::flow::queue_node<int> qeven_out(g);
tbb::flow::make_edge(qin,mf);
tbb::flow::make_edge(tbb::flow::output_port<0>(mf), qeven_out);
tbb::flow::make_edge(tbb::flow::output_port<1>(mf), qodd_out);
g.wait_for_all();
for( int ii = 0; ii < 2 ; ++ii) {
serial_fn_state0 = 0;
if(ii == 0) REMARK(" reset preds"); else REMARK(" 2nd");
qin.try_put(0);
BACKOFF_WAIT(serial_fn_state0 == 0, "timed out waiting for first put");
qin.try_put(1);
BACKOFF_WAIT((!my_test(mf)), "Timed out waiting");
ASSERT(my_test(mf), "fail second put test");
g.my_root_task->cancel_group_execution();
serial_fn_state0 = 2;
g.wait_for_all();
ASSERT(my_test(mf), "fail cancel group test");
if( ii == 1) {
REMARK(" rf_clear_edges");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(tbb::flow::output_port<0>(mf).my_successors.empty(), "output_port<0> not reset (rf_clear_edges)");
ASSERT(tbb::flow::output_port<1>(mf).my_successors.empty(), "output_port<1> not reset (rf_clear_edges)");
}
else
{
g.reset();
}
ASSERT(mf.my_predecessors.empty(), "edge didn't reset");
ASSERT((ii == 0 && !qin.my_successors.empty()) || (ii == 1 && qin.my_successors.empty()), "edge didn't reset");
}
REMARK(" done\n");
}

void
TestIndexerNode() {
tbb::flow::graph g;
typedef tbb::flow::indexer_node< int, int > indexernode_type;
indexernode_type inode(g);
REMARK("Testing indexer_node:");
tbb::flow::queue_node<indexernode_type::output_type> qout(g);
tbb::flow::make_edge(inode,qout);
g.wait_for_all();
ASSERT(!inode.my_successors.empty(), "successor of indexer_node missing");
g.reset();
ASSERT(!inode.my_successors.empty(), "successor of indexer_node missing after reset");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(inode.my_successors.empty(), "successor of indexer_node not removed by reset(rf_clear_edges)");
REMARK(" done\n");
}

template<typename Node>
void
TestScalarNode(const char *name) {
tbb::flow::graph g;
Node on(g);
tbb::flow::queue_node<int> qout(g);
REMARK("Testing %s:", name);
tbb::flow::make_edge(on,qout);
g.wait_for_all();
ASSERT(!on.my_successors.empty(), "edge not added");
g.reset();
ASSERT(!on.my_successors.empty(), "edge improperly removed");
g.reset(tbb::flow::rf_clear_edges);
ASSERT(on.my_successors.empty(), "edge not removed by reset(rf_clear_edges)");
REMARK(" done\n");
}

struct seq_body {
size_t operator()(const int &in) {
return size_t(in / 3);
}
};

void
TestSequencerNode() {
tbb::flow::graph g;
tbb::flow::sequencer_node<int> bnode(g, seq_body());
REMARK("Testing sequencer_node:");
tbb::flow::function_node<int> fnode(g, tbb::flow::serial, serial_fn_body<int>(serial_fn_state0));
REMARK("Testing sequencer_node:");
serial_fn_state0 = 0;  
REMARK(" make_edge");
tbb::flow::make_edge(bnode, fnode);
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after make_edge");
REMARK(" try_put");
bnode.try_put(0);  
BACKOFF_WAIT( serial_fn_state0 == 0, "timeout waiting for function_node");  
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after forwarding message");
serial_fn_state0 = 0;
g.wait_for_all();
REMARK(" remove_edge");
tbb::flow::remove_edge(bnode, fnode);
ASSERT(bnode.my_successors.empty(), "buffering node has a successor after remove_edge");
tbb::flow::join_node<tbb::flow::tuple<int,int>,tbb::flow::reserving> jnode(g);
tbb::flow::make_edge(bnode, tbb::flow::input_port<0>(jnode));  
g.wait_for_all();
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after attaching to join");
REMARK(" reverse");
bnode.try_put(3);  
g.wait_for_all();
ASSERT(bnode.my_successors.empty(), "buffering node has a successor after reserving");
REMARK(" reset()");
g.wait_for_all();
g.reset();  
ASSERT(!bnode.my_successors.empty(), "buffering node has no successor after reset()");
REMARK(" remove_edge");
g.reset(tbb::flow::rf_clear_edges);  
ASSERT(bnode.my_successors.empty(), "buffering node has a successor after reset(rf_clear_edges)");
ASSERT(fnode.my_predecessors.empty(), "buffering node reversed after reset(rf_clear_edges)");
REMARK("  done\n");
g.wait_for_all();
}

struct snode_body {
int max_cnt;
int my_cnt;
snode_body( const int &in) : max_cnt(in) { my_cnt = 0; }
#if TBB_DEPRECATED_INPUT_NODE_BODY
bool operator()(int &out) {
if(max_cnt <= my_cnt++) return false;
out = my_cnt;
return true;
}
#else
int operator()(tbb::flow_control &fc) {
if(max_cnt <= my_cnt++) {
fc.stop();
return int();
}
return my_cnt;
}
#endif
};

void
TestSourceNode() {
tbb::flow::graph g;
tbb::flow::input_node<int> sn(g, snode_body(4));
REMARK("Testing input_node:");
tbb::flow::queue_node<int> qin(g);
tbb::flow::join_node<tbb::flow::tuple<int,int>, tbb::flow::reserving> jn(g);
tbb::flow::queue_node<tbb::flow::tuple<int,int> > qout(g);

REMARK(" make_edges");
tbb::flow::make_edge(sn, tbb::flow::input_port<0>(jn));
tbb::flow::make_edge(qin, tbb::flow::input_port<1>(jn));
tbb::flow::make_edge(jn,qout);
ASSERT(!sn.my_successors.empty(), "source node has no successor after make_edge");
g.wait_for_all();
g.reset();
ASSERT(!sn.my_successors.empty(), "source node has no successor after reset");
g.wait_for_all();
g.reset(tbb::flow::rf_clear_edges);
ASSERT(sn.my_successors.empty(), "source node has successor after reset(rf_clear_edges)");
tbb::flow::make_edge(sn, tbb::flow::input_port<0>(jn));
tbb::flow::make_edge(qin, tbb::flow::input_port<1>(jn));
tbb::flow::make_edge(jn,qout);
g.wait_for_all();
REMARK(" activate");
sn.activate();  
REMARK(" wait1");
BACKOFF_WAIT( !sn.my_successors.empty(), "Timed out waiting for edge to reverse");
ASSERT(sn.my_successors.empty(), "source node has no successor after forwarding message");

g.wait_for_all();
g.reset();
ASSERT(!sn.my_successors.empty(), "input_node has no successors after reset");
ASSERT(tbb::flow::input_port<0>(jn).my_predecessors.empty(), "successor of input_node has pred after reset.");
REMARK(" done\n");
}

int TestMain() {

if(MinThread < 3) MinThread = 3;
tbb::task_scheduler_init init(MinThread);  

TestBufferingNode< tbb::flow::buffer_node<int> >("buffer_node");
TestBufferingNode< tbb::flow::priority_queue_node<int> >("priority_queue_node");
TestBufferingNode< tbb::flow::queue_node<int> >("queue_node");
TestSequencerNode();

TestMultifunctionNode<tbb::flow::rejecting>();
TestMultifunctionNode<tbb::flow::queueing>();
TestSourceNode();
TestContinueNode();
TestFunctionNode();

TestJoinNode();

TestLimiterNode();
TestIndexerNode();
TestSplitNode();
TestScalarNode<tbb::flow::broadcast_node<int> >("broadcast_node");
TestScalarNode<tbb::flow::overwrite_node<int> >("overwrite_node");
TestScalarNode<tbb::flow::write_once_node<int> >("write_once_node");

return Harness::Done;
}

