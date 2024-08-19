

#include "harness.h"
#include "tbb/flow_graph.h"
#include "tbb/task_scheduler_init.h"

#if defined(_MSC_VER) && _MSC_VER < 1600
#pragma warning (disable : 4503) 
#endif


const int Count = 300;
const int MaxPorts = 10;
const int MaxNSources = 5; 

std::vector<bool> flags;   

template<typename T>
class name_of {
public:
static const char* name() { return  "Unknown"; }
};
template<>
class name_of<int> {
public:
static const char* name() { return  "int"; }
};
template<>
class name_of<float> {
public:
static const char* name() { return  "float"; }
};
template<>
class name_of<double> {
public:
static const char* name() { return  "double"; }
};
template<>
class name_of<long> {
public:
static const char* name() { return  "long"; }
};
template<>
class name_of<short> {
public:
static const char* name() { return  "short"; }
};


template<int N>
struct tuple_helper {
template<typename TupleType>
static void set_element( TupleType &t, int i) {
tbb::flow::get<N-1>(t) = (typename tbb::flow::tuple_element<N-1,TupleType>::type)(i * (N+1));
tuple_helper<N-1>::set_element(t, i);
}
};

template<>
struct tuple_helper<1> {
template<typename TupleType>
static void set_element(TupleType &t, int i) {
tbb::flow::get<0>(t) = (typename tbb::flow::tuple_element<0,TupleType>::type)(i * 2);
}
};

template<typename TupleType>
class source_body {
typedef TupleType TT;
static const int N = tbb::flow::tuple_size<TT>::value;
int my_count;
int addend;
public:
source_body(int init_val, int addto) : my_count(init_val), addend(addto) { }
void operator=( const source_body& other) { my_count = other.my_count; addend = other.addend; }
bool operator()( TT &v) {
if(my_count >= Count) return false;
tuple_helper<N>::set_element(v, my_count);
my_count += addend;
return true;
}
};


template<int N, typename SType>
class makeSplit {
public:
static SType *create(tbb::flow::graph& g) {
SType *temp = new SType(g);
return temp;
}
static void destroy(SType *p) { delete p; }
};


static void* all_sink_nodes[MaxPorts];


template<int ELEM, typename SType>
class sink_node_helper {
public:
typedef typename SType::input_type TT;
typedef typename tbb::flow::tuple_element<ELEM-1,TT>::type IT;
typedef typename tbb::flow::queue_node<IT> my_sink_node_type;
static void print_parallel_remark() {
sink_node_helper<ELEM-1,SType>::print_parallel_remark();
REMARK(", %s", name_of<IT>::name());
}
static void print_serial_remark() {
sink_node_helper<ELEM-1,SType>::print_serial_remark();
REMARK(", %s", name_of<IT>::name());
}
static void add_sink_nodes(SType &my_split, tbb::flow::graph &g) {
my_sink_node_type *new_node = new my_sink_node_type(g);
tbb::flow::make_edge( tbb::flow::output_port<ELEM-1>(my_split) , *new_node);
all_sink_nodes[ELEM-1] = (void *)new_node;
sink_node_helper<ELEM-1, SType>::add_sink_nodes(my_split, g);
}

static void check_sink_values() {
my_sink_node_type *dp = reinterpret_cast<my_sink_node_type *>(all_sink_nodes[ELEM-1]);
for(int i = 0; i < Count; ++i) {
IT v;
ASSERT(dp->try_get(v), NULL);
flags[((int)v) / (ELEM+1)] = true;
}
for(int i = 0; i < Count; ++i) {
ASSERT(flags[i], NULL);
flags[i] = false;  
}
sink_node_helper<ELEM-1,SType>::check_sink_values();
}
static void remove_sink_nodes(SType& my_split) {
my_sink_node_type *dp = reinterpret_cast<my_sink_node_type *>(all_sink_nodes[ELEM-1]);
tbb::flow::remove_edge( tbb::flow::output_port<ELEM-1>(my_split) , *dp);
delete dp;
sink_node_helper<ELEM-1, SType>::remove_sink_nodes(my_split);
}
};

template<typename SType>
class sink_node_helper<1, SType> {
typedef typename SType::input_type TT;
typedef typename tbb::flow::tuple_element<0,TT>::type IT;
typedef typename tbb::flow::queue_node<IT> my_sink_node_type;
public:
static void print_parallel_remark() {
REMARK("Parallel test of split_node< %s", name_of<IT>::name());
}
static void print_serial_remark() {
REMARK("Serial test of split_node< %s", name_of<IT>::name());
}
static void add_sink_nodes(SType &my_split, tbb::flow::graph &g) {
my_sink_node_type *new_node = new my_sink_node_type(g);
tbb::flow::make_edge( tbb::flow::output_port<0>(my_split) , *new_node);
all_sink_nodes[0] = (void *)new_node;
}
static void check_sink_values() {
my_sink_node_type *dp = reinterpret_cast<my_sink_node_type *>(all_sink_nodes[0]);
for(int i = 0; i < Count; ++i) {
IT v;
ASSERT(dp->try_get(v), NULL);
flags[((int)v) / 2] = true;
}
for(int i = 0; i < Count; ++i) {
ASSERT(flags[i], NULL);
flags[i] = false;  
}
}
static void remove_sink_nodes(SType& my_split) {
my_sink_node_type *dp = reinterpret_cast<my_sink_node_type *>(all_sink_nodes[0]);
tbb::flow::remove_edge( tbb::flow::output_port<0>(my_split) , *dp);
delete dp;
}
};

template<typename SType>
class parallel_test {
public:
typedef typename SType::input_type TType;
typedef tbb::flow::source_node<TType> source_type;
static const int N = tbb::flow::tuple_size<TType>::value;
static void test() {
TType v;
source_type* all_source_nodes[MaxNSources];
sink_node_helper<N,SType>::print_parallel_remark();
REMARK(" >\n");
for(int i=0; i < MaxPorts; ++i) {
all_sink_nodes[i] = NULL;
}
for(int nInputs = 1; nInputs <= MaxNSources; ++nInputs) {
tbb::flow::graph g;
SType* my_split = makeSplit<N,SType>::create(g);

sink_node_helper<N, SType>::add_sink_nodes((*my_split), g);

for(int i = 0; i < nInputs; ++i) {
source_type *s = new source_type(g, source_body<TType>(i, nInputs) );
tbb::flow::make_edge(*s, *my_split);
all_source_nodes[i] = s;
}

g.wait_for_all();

sink_node_helper<N, SType>::check_sink_values();

sink_node_helper<N, SType>::remove_sink_nodes(*my_split);
for(int i = 0; i < nInputs; ++i) {
delete all_source_nodes[i];
}
makeSplit<N,SType>::destroy(my_split);
}
}
};


template<typename SType>
void test_one_serial( SType &my_split, tbb::flow::graph &g) {
typedef typename SType::input_type TType;
static const int TUPLE_SIZE = tbb::flow::tuple_size<TType>::value;
sink_node_helper<TUPLE_SIZE, SType>::add_sink_nodes(my_split,g);
typedef TType q3_input_type;
tbb::flow::queue_node< q3_input_type >  q3(g);

tbb::flow::make_edge( q3, my_split );

flags.clear();
for (int i = 0; i < Count; ++i ) {
TType v;
tuple_helper<TUPLE_SIZE>::set_element(v, i);
ASSERT(my_split.try_put(v), NULL);
flags.push_back(false);
}

g.wait_for_all();

sink_node_helper<TUPLE_SIZE,SType>::check_sink_values();

sink_node_helper<TUPLE_SIZE, SType>::remove_sink_nodes(my_split);

}

template<typename SType>
class serial_test {
typedef typename SType::input_type TType;
static const int TUPLE_SIZE = tbb::flow::tuple_size<TType>::value;
static const int ELEMS = 3;
public:
static void test() {
tbb::flow::graph g;
flags.reserve(Count);
SType* my_split = makeSplit<TUPLE_SIZE,SType>::create(g);
sink_node_helper<TUPLE_SIZE, SType>::print_serial_remark(); REMARK(" >\n");

test_one_serial<SType>( *my_split, g);
std::vector<SType>split_vector(ELEMS, *my_split);
makeSplit<TUPLE_SIZE,SType>::destroy(my_split);


for(int e = 0; e < ELEMS; ++e) {  
test_one_serial<SType>( split_vector[e], g);
}
}

}; 

template<
template<typename> class TestType,  
typename TupleType >                               
struct generate_test {
typedef tbb::flow::split_node<TupleType> split_node_type;
static void do_test() {
TestType<split_node_type>::test();
}
}; 

int TestMain() {
#if __TBB_USE_TBB_TUPLE
REMARK("  Using TBB tuple\n");
#else
REMARK("  Using platform tuple\n");
#endif
for (int p = 0; p < 2; ++p) {
generate_test<serial_test, tbb::flow::tuple<float, double> >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 4
generate_test<serial_test, tbb::flow::tuple<float, double, int, long> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 6
generate_test<serial_test, tbb::flow::tuple<double, double, int, long, int, short> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 8
generate_test<serial_test, tbb::flow::tuple<float, double, double, double, float, int, float, long> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 10
generate_test<serial_test, tbb::flow::tuple<float, double, int, double, double, float, long, int, float, long> >::do_test();
#endif
generate_test<parallel_test, tbb::flow::tuple<float, double> >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 3
generate_test<parallel_test, tbb::flow::tuple<float, int, long> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 5
generate_test<parallel_test, tbb::flow::tuple<double, double, int, int, short> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
generate_test<parallel_test, tbb::flow::tuple<float, int, double, float, long, float, long> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 9
generate_test<parallel_test, tbb::flow::tuple<float, double, int, double, double, long, int, float, long> >::do_test();
#endif
}
return Harness::Done;
}
