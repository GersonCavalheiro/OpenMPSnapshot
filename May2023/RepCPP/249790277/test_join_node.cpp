

#if _MSC_VER
#if __INTEL_COMPILER
#pragma warning( disable: 2586 )
#else
#pragma warning( disable: 4503 )
#endif
#endif

#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1
#include "harness.h"
#include "harness_defs.h"
#include "tbb/atomic.h"
#include "harness_checktype.h"

#include "tbb/flow_graph.h"
#include "tbb/task_scheduler_init.h"

#define __TBB_MIC_OFFLOAD_TEST_COMPILATION_BROKEN __TBB_MIC_OFFLOAD

const char *names[] = {
"Adam", "Bruce", "Charles", "Daniel", "Evan", "Frederich", "George", "Hiram", "Ichabod",
"John", "Kevin", "Leonard", "Michael", "Ned", "Olin", "Paul", "Quentin", "Ralph", "Steven",
"Thomas", "Ulysses", "Victor", "Walter", "Xerxes", "Yitzhak", "Zebediah", "Anne", "Bethany",
"Clarisse", "Dorothy", "Erin", "Fatima", "Gabrielle", "Helen", "Irene", "Jacqueline",
"Katherine", "Lana", "Marilyn", "Noelle", "Okiilani", "Pauline", "Querida", "Rose", "Sybil",
"Tatiana", "Umiko", "Victoria", "Wilma", "Xena", "Yolanda", "Zoe", "Algernon", "Benjamin",
"Caleb", "Dylan", "Ezra", "Felix", "Gabriel", "Henry", "Issac", "Jasper", "Keifer",
"Lincoln", "Milo", "Nathaniel", "Owen", "Peter", "Quincy", "Ronan", "Silas", "Theodore",
"Uriah", "Vincent", "Wilbur", "Xavier", "Yoda", "Zachary", "Amelia", "Brielle", "Charlotte",
"Daphne", "Emma", "Fiona", "Grace", "Hazel", "Isla", "Juliet", "Keira", "Lily", "Mia",
"Nora", "Olivia", "Penelope", "Quintana", "Ruby", "Sophia", "Tessa", "Ursula", "Violet",
"Willow", "Xanthe", "Yvonne", "ZsaZsa", "Asher", "Bennett", "Connor", "Dominic", "Ethan",
"Finn", "Grayson", "Hudson", "Ian", "Jackson", "Kent", "Liam", "Matthew", "Noah", "Oliver",
"Parker", "Quinn", "Rhys", "Sebastian", "Taylor", "Umberto", "Vito", "William", "Xanto",
"Yogi", "Zane", "Ava", "Brenda", "Chloe", "Delilah", "Ella", "Felicity", "Genevieve",
"Hannah", "Isabella", "Josephine", "Kacie", "Lucy", "Madeline", "Natalie", "Octavia",
"Piper", "Qismah", "Rosalie", "Scarlett", "Tanya", "Uta", "Vivian", "Wendy", "Xola",
"Yaritza", "Zanthe" };

static const int NameCnt = sizeof(names) / sizeof(char *);

template<typename K>
struct index_to_key {
K operator()(const int indx){
return (K)(3*indx+1);
}
};

template<>
struct index_to_key<std::string>{
std::string operator()(const int indx){
return std::string(names[indx % NameCnt]);
}
};

template<typename K>
struct K_deref {
typedef K type;
};

template<typename K>
struct K_deref<K&> {
typedef K type;
};

template<typename K, typename V>
struct MyKeyFirst {
K my_key;
V my_value;
MyKeyFirst(int i=0, int v=0 ) : my_key(index_to_key<K>()(i)), my_value((V)v){
}
void print_val() const {
REMARK("MyKeyFirst{");print_my_value(my_key); REMARK(","); print_my_value(my_value);REMARK("}");
}
operator int() const { return (int)my_value; }
};

template<typename K, typename V>
struct MyKeySecond {
V my_value;
K my_key;
MyKeySecond(int i=0, int v = 0) : my_value((V)v), my_key(index_to_key<K>()(i)) {
}
void print_val() const {
REMARK("MyKeySecond{");print_my_value(my_key); REMARK(","); print_my_value(my_value);REMARK("}");
}
operator int() const { return (int)my_value; }
};

template<typename K, typename V>
struct MyMessageKeyWithoutKey {
V my_value;
K my_message_key;
MyMessageKeyWithoutKey(int i=0, int v = 0) : my_value((V)v), my_message_key(index_to_key<K>()(i)) {
}
void print_val() const {
REMARK("MyMessageKeyWithoutKey{");print_my_value(my_message_key); REMARK(","); print_my_value(my_value);REMARK("}");
}
operator int() const { return (int)my_value; }
const K& key() const {
return my_message_key;
}
};

template<typename K, typename V>
struct MyMessageKeyWithBrokenKey {
V my_value;
K my_key;
K my_message_key;
MyMessageKeyWithBrokenKey(int i=0, int v = 0) : my_value((V)v), my_key(), my_message_key(index_to_key<K>()(i)) {
}
void print_val() const {
REMARK("MyMessageKeyWithBrokenKey{");print_my_value(my_message_key); REMARK(","); print_my_value(my_value);REMARK("}");
}
operator int() const { return (int)my_value; }
const K& key() const {
return my_message_key;
}

};

template<typename K, typename V>
struct MyKeyWithBrokenMessageKey {
V my_value;
K my_key;
MyKeyWithBrokenMessageKey(int i=0, int v = 0) : my_value((V)v), my_key(index_to_key<K>()(i)) {
}
void print_val() const {
REMARK("MyKeyWithBrokenMessageKey{");print_my_value(my_key); REMARK(","); print_my_value(my_value);REMARK("}");
}
operator int() const { return (int)my_value; }
K key() const {
ASSERT(false, "The method should never be called");
return K();
}
};

template<typename K, typename V>
struct MyMessageKeyWithoutKeyMethod {
V my_value;
K my_message_key;
MyMessageKeyWithoutKeyMethod(int i=0, int v = 0) : my_value((V)v), my_message_key(index_to_key<K>()(i)) {
}
void print_val() const {
REMARK("MyMessageKeyWithoutKeyMethod{");print_my_value(my_message_key); REMARK(","); print_my_value(my_value);REMARK("}");
}
operator int() const { return (int)my_value; }
#if __TBB_COMPLICATED_ADL_BROKEN
const K& key() const { return my_message_key; }
#endif
};

template <typename K, typename V>
K key_from_message(const MyMessageKeyWithoutKeyMethod<typename tbb::internal::strip<K>::type, V> &m) {
return m.my_message_key;
}


template<typename TT, size_t INDEX>
struct make_thingie {
TT operator()(int const &i) {
return TT(i * (INDEX+1));
}
};

template<template <typename, typename> class T, typename K, typename V, size_t INDEX>
struct make_thingie<T<K,V>, INDEX> {
T<K,V> operator()(int const &i) {
return T<K,V>(i, i*(INDEX+1));
}
};

template<typename T>
struct cast_from {
static int my_int_val(T const &i) { return (int)i; }
};

template<typename K, typename V>
struct cast_from<MyKeyFirst<K,V> > {
static int my_int_val(MyKeyFirst<K,V> const &i) { return (int)(i.my_value); }
};

template<typename K, typename V>
struct cast_from<MyKeySecond<K,V> > {
static int my_int_val(MyKeySecond<K,V> const &i) { return (int)(i.my_value); }
};

template<typename T>
void print_my_value(T const &i) {
REMARK(" %d ", cast_from<T>::my_int_val(i));
}

template<typename K, typename V>
void print_my_value(MyKeyFirst<K,V> const &i) {
i.print_val();
}

template<typename K, typename V>
void print_my_value(MyKeySecond<K,V> const &i) {
i.print_val();
}

template<>
void print_my_value(std::string const &i) {
REMARK("\"%s\"", i.c_str());
}


template<class K, typename V>
struct my_struct_key{
K operator()(const V& mv) {
return mv.my_key;
}
};

template<class K, typename V>
struct my_struct_key<K&, V>{
const K& operator()(const V& mv) {
return const_cast<const K&>(mv.my_key);
}
};

using tbb::internal::is_ref;

template<class K, class V> struct VtoKFB {
typedef tbb::flow::interface9::internal::type_to_key_function_body<V, K> type;
};

template<typename K> struct make_hash_compare{ typedef typename tbb::tbb_hash_compare<K> type;};

template<typename K, class V>
void hash_buffer_test(const char *sname) {
typedef typename K_deref<K>::type KnoR;
tbb::flow::interface9::internal::hash_buffer<
K,
V,
typename VtoKFB<K,V>::type,
tbb::tbb_hash_compare<KnoR>
> my_hash_buffer;
const bool k_is_ref = is_ref<K>::value;
typedef tbb::flow::interface9::internal::type_to_key_function_body_leaf<
V, K, my_struct_key<K,V> > my_func_body_type;
typename VtoKFB<K,V>::type *kp = new my_func_body_type(my_struct_key<K,V>());
my_hash_buffer.set_key_func(kp);
REMARK("Running hash_buffer test on %s; is ref == %s\n", sname, k_is_ref ? "true":"false");
V mv1, mv0;
bool res;
for(int cnt = 0; cnt < 2; ++cnt) {
for( int i = 0; i < 50; ++i) {
KnoR kk = index_to_key<KnoR>()(i);
mv1.my_key = kk;
mv1.my_value = 0.5*i;
res = my_hash_buffer.find_with_key(kk, mv0);
ASSERT(!res, "Found non-inserted item");
res = my_hash_buffer.insert_with_key(mv1);
ASSERT(res, "insert failed");
res = my_hash_buffer.find_with_key(kk, mv0);
ASSERT(res, "not found after insert");
ASSERT(mv0.my_value == mv1.my_value, "result not correct");
}
for( int i = 49; i >= 0; --i) {
KnoR kk = index_to_key<KnoR>()(i);
double value = 0.5*i;
res = my_hash_buffer.find_with_key(kk, mv0);
ASSERT(res, "find failed");
ASSERT(mv0.my_value == value, "result not correct");
}
for(int i = 0; i < 50; i += 3) {
KnoR kk = index_to_key<KnoR>()(i);
my_hash_buffer.delete_with_key(kk);
res = my_hash_buffer.find_with_key(kk, mv0);
ASSERT(!res, "Found deleted item");
}
for(int i = 0; i < 50; ++i) {
KnoR kk = index_to_key<KnoR>()(i);
double value = 0.5*i;
if(i%3 == 0) {
res = my_hash_buffer.find_with_key(kk, mv0);
ASSERT(!res, "found an item that was previously deleted");
}
else {
res = my_hash_buffer.find_with_key(kk, mv0);
ASSERT(res, "find failed");
ASSERT(mv0.my_value == value, "result not correct");
}
}
for(int i = 0; i < 50; ++i) {
KnoR kk = index_to_key<KnoR>()(i);
double value = 1.5*i;
mv1.my_key = kk;
mv1.my_value = value;
res = my_hash_buffer.insert_with_key(mv1);
if(i%3 == 0) {
ASSERT(res, "didn't insert in empty slot");
}
else {
ASSERT(!res, "slot was empty on insert");
}
}
for(int i = 0; i < 50; ++i) {
KnoR kk = index_to_key<KnoR>()(i);
my_hash_buffer.delete_with_key(kk);
res = my_hash_buffer.find_with_key(kk, mv0);
ASSERT(!res, "Found deleted item");
}
}  
}

void
TestTaggedBuffers() {
hash_buffer_test<int, MyKeyFirst<int,double> >("MyKeyFirst<int,double>");
hash_buffer_test<int&, MyKeyFirst<int,double> >("MyKeyFirst<int,double> with int&");
hash_buffer_test<int, MyKeySecond<int,double> >("MyKeySecond<int,double>");

hash_buffer_test<std::string, MyKeyFirst<std::string,double> >("MyKeyFirst<std::string,double>");
hash_buffer_test<std::string&, MyKeySecond<std::string,double> >("MyKeySecond<std::string,double> with std::string&");
}

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
template< typename T, typename NODE_TYPE >
class test_join_base_extract : NoAssign {
protected:
typedef typename NODE_TYPE::output_type tuple_t;
typedef tbb::flow::queue_node<T> in_queue_t;
typedef tbb::flow::queue_node<tuple_t> out_queue_t;

tbb::flow::graph &g;
in_queue_t &in0;
in_queue_t &in1;
in_queue_t &in2;
NODE_TYPE &middle;
out_queue_t &out0;
out_queue_t &out1;
in_queue_t *ins[3];
out_queue_t *outs[2];
typename in_queue_t::successor_type *ms_p0_ptr;
typename in_queue_t::successor_type *ms_p1_ptr;
typename out_queue_t::predecessor_type *mp_ptr;
typename in_queue_t::predecessor_list_type in0_p_list;
typename in_queue_t::successor_list_type in0_s_list;
typename in_queue_t::predecessor_list_type in1_p_list;
typename in_queue_t::successor_list_type in1_s_list;
typename in_queue_t::predecessor_list_type in2_p_list;
typename in_queue_t::successor_list_type in2_s_list;
typename out_queue_t::predecessor_list_type out0_p_list;
typename out_queue_t::successor_list_type out0_s_list;
typename out_queue_t::predecessor_list_type out1_p_list;
typename out_queue_t::successor_list_type out1_s_list;
typename in_queue_t::predecessor_list_type mp0_list;
typename in_queue_t::predecessor_list_type mp1_list;
typename out_queue_t::successor_list_type ms_list;

virtual void set_up_lists() {
in0_p_list.clear();
in0_s_list.clear();
in1_p_list.clear();
in1_s_list.clear();
in2_p_list.clear();
in2_s_list.clear();
out0_p_list.clear();
out0_s_list.clear();
out1_p_list.clear();
out1_s_list.clear();
mp0_list.clear();
mp1_list.clear();
ms_list.clear();

in0.copy_predecessors(in0_p_list);
in0.copy_successors(in0_s_list);
in1.copy_predecessors(in1_p_list);
in1.copy_successors(in1_s_list);
in2.copy_predecessors(in2_p_list);
in2.copy_successors(in2_s_list);
tbb::flow::input_port<0>(middle).copy_predecessors(mp0_list);
tbb::flow::input_port<1>(middle).copy_predecessors(mp1_list);
middle.copy_successors(ms_list);
out0.copy_predecessors(out0_p_list);
out0.copy_successors(out0_s_list);
out1.copy_predecessors(out1_p_list);
out1.copy_successors(out1_s_list);
}

void check_tuple( T &r, tuple_t &v ) {
T t0 = tbb::flow::get<0>(v);
T t1 = tbb::flow::get<1>(v);
ASSERT( (t0 == 1 || t0 == 2) && (t0&r) == 0, "duplicate value" );
r |= t0;
ASSERT( (t1 == 4 || t1 == 8) && (t1&r) == 0, "duplicate value" );
r |= t1;
}

void make_and_validate_full_graph() {







tbb::flow::make_edge( in0, tbb::flow::input_port<0>(middle) );
tbb::flow::make_edge( in1, tbb::flow::input_port<0>(middle) );
tbb::flow::make_edge( in2, tbb::flow::input_port<1>(middle) );
tbb::flow::make_edge( middle, out0 );
tbb::flow::make_edge( middle, out1 );

set_up_lists();

ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in0.successor_count() == 1 && in0_s_list.size() == 1 && *(in0_s_list.begin()) == ms_p0_ptr, "expected 1 successor" );
ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in1.successor_count() == 1 && in1_s_list.size() == 1 && *(in1_s_list.begin()) == ms_p0_ptr, "expected 1 successor" );
ASSERT( in2.predecessor_count() == 0 && in2_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in2.successor_count() == 1 && in2_s_list.size() == 1 && *(in2_s_list.begin()) == ms_p1_ptr, "expected 1 successor" );
ASSERT( tbb::flow::input_port<0>(middle).predecessor_count() == 2 && mp0_list.size() == 2, "expected 2 predecessors" );
ASSERT( tbb::flow::input_port<1>(middle).predecessor_count() == 1 && mp1_list.size() == 1, "expected 1 predecessors" );
ASSERT( middle.successor_count() == 2 && ms_list.size() == 2, "expected 2 successors" );
ASSERT( out0.predecessor_count() == 1 && out0_p_list.size() == 1 && *(out0_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
ASSERT( out1.predecessor_count() == 1 && out1_p_list.size() == 1 && *(out1_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

typename in_queue_t::predecessor_list_type::iterator mp0_list_iter = mp0_list.begin(); ++mp0_list_iter;
int first_pred = *(mp0_list.begin()) == ins[0] ? 0 : ( *(mp0_list.begin()) == ins[1] ? 1 : -1 );
int second_pred = *mp0_list_iter == ins[0] ? 0 : ( *mp0_list_iter == ins[1] ? 1 : -1 );
ASSERT( first_pred != -1 && second_pred != -1 && first_pred != second_pred, "bad predecessor(s) for middle port 0" );

ASSERT( *(mp1_list.begin()) == ins[2], "bad predecessor for middle port 1" );

typename out_queue_t::successor_list_type::iterator ms_list_iter = ms_list.begin(); ++ms_list_iter;
int first_succ = *(ms_list.begin()) == outs[0] ? 0 : ( *(ms_list.begin()) == outs[1] ? 1 : -1 );
int second_succ = *ms_list_iter == outs[0] ? 0 : ( *ms_list_iter == outs[1] ? 1 : -1 );
ASSERT( first_succ != -1 && second_succ != -1 && first_succ != second_succ, "bad successor(s) for middle" );

in0.try_put(1);
in1.try_put(2);
in2.try_put(8);
in2.try_put(4);
g.wait_for_all();

T v_in;
tuple_t v;

ASSERT( in0.try_get(v_in) == false, "buffer should not have a value" );
ASSERT( in1.try_get(v_in) == false, "buffer should not have a value" );
ASSERT( in1.try_get(v_in) == false, "buffer should not have a value" );
ASSERT( in2.try_get(v_in) == false, "buffer should not have a value" );
ASSERT( in2.try_get(v_in) == false, "buffer should not have a value" );

T r = 0;
while ( out0.try_get(v) ) {
check_tuple(r,v);
g.wait_for_all();
}
ASSERT( r == 15, "not all values received" );

r = 0;
while ( out1.try_get(v) ) {
check_tuple(r,v);
g.wait_for_all();
}
ASSERT( r == 15, "not all values received" );
g.wait_for_all();
}

void validate_partial_graph() {







set_up_lists();

ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in0.successor_count() == 0 && in0_s_list.size() == 0, "expected 0 successors" );
ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in1.successor_count() == 1 && in1_s_list.size() == 1 && *(in1_s_list.begin()) == ms_p0_ptr, "expected 1 successor" );
ASSERT( in2.predecessor_count() == 0 && in2_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in2.successor_count() == 1 && in2_s_list.size() == 1 && *(in2_s_list.begin()) == ms_p1_ptr, "expected 1 successor" );
ASSERT( tbb::flow::input_port<0>(middle).predecessor_count() == 1 && mp0_list.size() == 1 && *(mp0_list.begin()) == ins[1], "expected 1 predecessor" );
ASSERT( tbb::flow::input_port<1>(middle).predecessor_count() == 1 && mp1_list.size() == 1 && *(mp1_list.begin()) == ins[2], "expected 1 predecessor" );
ASSERT( middle.successor_count() == 1 && ms_list.size() == 1 && *(ms_list.begin()) == outs[1], "expected 1 successor" );
ASSERT( out0.predecessor_count() == 0 && out0_p_list.size() == 0, "expected 1 predecessor" );
ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
ASSERT( out1.predecessor_count() == 1 && out1_p_list.size() == 1 && *(out1_p_list.begin()) == mp_ptr, "expected 1 predecessor" );
ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

in0.try_put(1);
in1.try_put(2);
in2.try_put(8);
in2.try_put(4);
g.wait_for_all();

T v_in;
tuple_t v;

ASSERT( in0.try_get(v_in) == true && v_in == 1, "buffer should have a value of 1" );
ASSERT( in1.try_get(v_in) == false, "buffer should not have a value" );
ASSERT( out0.try_get(v) == false, "buffer should not have a value" );
ASSERT( out1.try_get(v) == true && tbb::flow::get<0>(v) == 2 && tbb::flow::get<1>(v) == 8, "buffer should have a value of < 2, 8 >" );
ASSERT( in0.try_get(v_in) == false, "buffer should not have a value" );
g.wait_for_all();
g.reset();  
}

void validate_empty_graph() {







set_up_lists();

ASSERT( in0.predecessor_count() == 0 && in0_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in0.successor_count() == 0 && in0_s_list.size() == 0, "expected 0 successors" );
ASSERT( in1.predecessor_count() == 0 && in1_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in1.successor_count() == 0 && in1_s_list.size() == 0, "expected 0 successors" );
ASSERT( in2.predecessor_count() == 0 && in2_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( in2.successor_count() == 0 && in2_s_list.size() == 0, "expected 0 successors" );
ASSERT( tbb::flow::input_port<0>(middle).predecessor_count() == 0 && mp0_list.size() == 0, "expected 0 predecessors" );
ASSERT( tbb::flow::input_port<1>(middle).predecessor_count() == 0 && mp1_list.size() == 0, "expected 0 predecessors" );
ASSERT( middle.successor_count() == 0 && ms_list.size() == 0, "expected 0 successors" );
ASSERT( out0.predecessor_count() == 0 && out0_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( out0.successor_count() == 0 && out0_s_list.size() == 0, "expected 0 successors" );
ASSERT( out1.predecessor_count() == 0 && out1_p_list.size() == 0, "expected 0 predecessors" );
ASSERT( out1.successor_count() == 0 && out1_s_list.size() == 0, "expected 0 successors" );

in0.try_put(1);
in1.try_put(2);
in2.try_put(8);
in2.try_put(4);
g.wait_for_all();

T v_in;
tuple_t v;

ASSERT( in0.try_get(v_in) == true && v_in == 1, "buffer should have a value of 1" );
ASSERT( in1.try_get(v_in) == true && v_in == 2, "buffer should have a value of 2" );
ASSERT( in2.try_get(v_in) == true && v_in == 8, "buffer should have a value of 8" );
ASSERT( in2.try_get(v_in) == true && v_in == 4, "buffer should have a value of 4" );
ASSERT( out0.try_get(v) == false, "buffer should not have a value" );
ASSERT( out1.try_get(v) == false, "buffer should not have a value" );
g.wait_for_all();
g.reset(); 
}

public:

test_join_base_extract(tbb::flow::graph &_g, in_queue_t &_in0, in_queue_t &_in1, in_queue_t &_in2, NODE_TYPE &m, out_queue_t &_out0, out_queue_t &_out1) :
g(_g), in0(_in0), in1(_in1), in2(_in2), middle(m), out0(_out0), out1(_out1) {
ins[0] = &in0;
ins[1] = &in1;
ins[2] = &in2;
outs[0] = &out0;
outs[1] = &out1;
ms_p0_ptr = static_cast< typename in_queue_t::successor_type * >(&tbb::flow::input_port<0>(middle));
ms_p1_ptr = static_cast< typename in_queue_t::successor_type * >(&tbb::flow::input_port<1>(middle));
mp_ptr = static_cast< typename out_queue_t::predecessor_type *>(&middle);
}

virtual ~test_join_base_extract() {}

void run_tests() {
REMARK("full graph\n");
make_and_validate_full_graph();

in0.extract();
out0.extract();
REMARK("partial graph\n");
validate_partial_graph();

in1.extract();
in2.extract();
out1.extract();
REMARK("empty graph\n");
validate_empty_graph();

REMARK("full graph\n");
make_and_validate_full_graph();

middle.extract();
REMARK("empty graph\n");
validate_empty_graph();

REMARK("full graph\n");
make_and_validate_full_graph();

in0.extract();
in1.extract();
in2.extract();
middle.extract();
REMARK("empty graph\n");
validate_empty_graph();

REMARK("full graph\n");
make_and_validate_full_graph();

out0.extract();
out1.extract();
middle.extract();
REMARK("empty graph\n");
validate_empty_graph();

REMARK("full graph\n");
make_and_validate_full_graph();
}
};

template< typename T, typename NODE_TYPE >
class test_join_extract : public test_join_base_extract< T, NODE_TYPE > {
protected:
typedef typename NODE_TYPE::output_type tuple_t;
typedef tbb::flow::queue_node<T> in_queue_t;
typedef tbb::flow::queue_node<tuple_t> out_queue_t;

tbb::flow::graph my_g;
in_queue_t my_in0;
in_queue_t my_in1;
in_queue_t my_in2;
NODE_TYPE my_middle;
out_queue_t my_out0;
out_queue_t my_out1;

public:
test_join_extract() : test_join_base_extract<T, NODE_TYPE>( my_g, my_in0, my_in1, my_in2, my_middle, my_out0, my_out1 ),
my_in0(my_g), my_in1(my_g), my_in2(my_g), my_middle(my_g), my_out0(my_g), my_out1(my_g) { }
};

template< typename T >
class test_join_extract<T, tbb::flow::join_node< tbb::flow::tuple<T,T>, tbb::flow::tag_matching> > :
public test_join_base_extract< T, tbb::flow::join_node< tbb::flow::tuple<T,T>, tbb::flow::tag_matching> > {
protected:
typedef tbb::flow::join_node< tbb::flow::tuple<T,T>, tbb::flow::tag_matching> my_node_t;

typedef typename my_node_t::output_type tuple_t;
typedef tbb::flow::queue_node<T> in_queue_t;
typedef tbb::flow::queue_node<tuple_t> out_queue_t;

tbb::flow::graph my_g;
in_queue_t my_in0;
in_queue_t my_in1;
in_queue_t my_in2;
my_node_t my_middle;
out_queue_t my_out0;
out_queue_t my_out1;
struct tag_match_0 { size_t operator()(T v) { return v; } };
struct tag_match_1 { size_t operator()(T v) { return v/4; } };
public:
test_join_extract() : test_join_base_extract<T, my_node_t>( my_g, my_in0, my_in1, my_in2, my_middle, my_out0, my_out1 ),
my_in0(my_g), my_in1(my_g), my_in2(my_g), my_middle(my_g, tag_match_0(), tag_match_1()), my_out0(my_g), my_out1(my_g) { }
};
#endif

struct threebyte {
unsigned char b1;
unsigned char b2;
unsigned char b3;
threebyte(int i=0) {
b1 = (unsigned char)(i & 0xFF);
b2 = (unsigned char)((i >> 8) & 0xFF);
b3 = (unsigned char)((i >> 16) & 0xFF);
}
threebyte(const threebyte &other) : b1(other.b1), b2(other.b2), b3(other.b3) { }
operator int() const { return (int)(b1 + (b2 << 8) + (b3 << 16)); }
};

const int Count = 150;

const int Recirc_count = 1000;  
const int MaxPorts = 10;
const int MaxNSources = 5; 
bool outputCheck[MaxPorts][Count];  

void
check_outputCheck( int nUsed, int maxCnt) {
for(int i=0; i < nUsed; ++i) {
for( int j = 0; j < maxCnt; ++j) {
ASSERT(outputCheck[i][j], NULL);
}
}
}

void
reset_outputCheck( int nUsed, int maxCnt) {
for(int i=0; i < nUsed; ++i) {
for( int j = 0; j < maxCnt; ++j) {
outputCheck[i][j] = false;
}
}
}

template<typename T>
class name_of {
public:
static const char* name() { return  "Unknown"; }
};
template<typename T>
class name_of<check_type<T> > {
public:
static const char* name() { return "checktype"; }
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
template<>
class name_of<threebyte> {
public:
static const char* name() {return "threebyte"; }
};
template<>
class name_of<std::string> {
public:
static const char* name() {return "std::string"; }
};
template<typename K,typename V>
class name_of<MyKeyFirst<K,V> > {
public:
static const char* name() {return "MyKeyFirst<K,V>"; }
};
template<typename K,typename V>
class name_of<MyKeySecond<K,V> > {
public:
static const char* name() {return "MyKeySecond<K,V>"; }
};

template <typename K, typename KHash = tbb::tbb_hash_compare<typename tbb::internal::strip<K>::type > >
struct message_based_key_matching {};

template<class JP>
struct is_key_matching_join {
static const bool value = false;
typedef int key_type;  
};
template<class K, class KHash>
struct is_key_matching_join<tbb::flow::key_matching<K,KHash> > {
static const bool value = true;
typedef K key_type;
};
template<class K, class KHash>
struct is_key_matching_join<message_based_key_matching<K,KHash> > {
static const bool value = true;
typedef K key_type;
};

template<typename TT>
class recirc_func_body {
TT my_mult;
public:
typedef tbb::flow::tuple<int, tbb::flow::continue_msg> input_type;
recirc_func_body(TT multiplier ) : my_mult(multiplier) {}
recirc_func_body(const recirc_func_body &other) : my_mult(other.my_mult) { }
void operator=( const recirc_func_body &other) { my_mult = other.my_mult; }
TT operator()(const input_type &v) {
return TT(tbb::flow::get<0>(v)) * my_mult;
}
};

static int input_count;  
static tbb::atomic<int> output_count;

class recirc_source_node_body {
public:
bool operator()(tbb::flow::continue_msg &v ) {
--input_count;
v = tbb::flow::continue_msg();
return 0 <= input_count;
}
};

template<typename TT, size_t INDEX>
class source_body {
int my_count;
int addend;
public:
source_body(int init_val, int addto) : my_count(init_val), addend(addto) { }
void operator=( const source_body& other) {my_count=other.my_count; addend=other.addend;}
bool operator()( TT &v) {
int lc = my_count;
v = make_thingie<TT, INDEX>()(my_count);
my_count += addend;
return lc < Count;
}
};

template<typename TT>
class tag_func {
TT my_mult;
public:
tag_func(TT multiplier) : my_mult(multiplier) { }
void operator=( const tag_func& other){my_mult = other.my_mult;}
tbb::flow::tag_value operator()( TT v) {
tbb::flow::tag_value t = tbb::flow::tag_value(v / my_mult);
return t;
}
};

template <class JP>
struct filter_out_message_based_key_matching {
typedef JP policy;
};

template <typename K, typename KHash>
struct filter_out_message_based_key_matching<message_based_key_matching<K, KHash> > {
typedef tbb::flow::key_matching<K, KHash> policy;
};


template<int N, typename JType, class JP>
class makeJoin {
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType, typename K, typename KHash>
class makeJoin<2,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<2,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

#if MAX_TUPLE_TEST_SIZE >= 3
template<typename JType, typename K, typename KHash>
class makeJoin<3,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<3,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

#endif
#if MAX_TUPLE_TEST_SIZE >= 4

template<typename JType, typename K, typename KHash>
class makeJoin<4,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>(),
my_struct_key<K,T3>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<4,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4)),
tag_func<T3>(T3(5))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

#endif
#if MAX_TUPLE_TEST_SIZE >= 5
template<typename JType, typename K, typename KHash>
class makeJoin<5,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>(),
my_struct_key<K,T3>(),
my_struct_key<K,T4>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<5,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4)),
tag_func<T3>(T3(5)),
tag_func<T4>(T4(6))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};
#endif
#if MAX_TUPLE_TEST_SIZE >= 6
template<typename JType, typename K, typename KHash>
class makeJoin<6,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>(),
my_struct_key<K,T3>(),
my_struct_key<K,T4>(),
my_struct_key<K,T5>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<6,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4)),
tag_func<T3>(T3(5)),
tag_func<T4>(T4(6)),
tag_func<T5>(T5(7))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};
#endif

#if MAX_TUPLE_TEST_SIZE >= 7
template<typename JType, typename K, typename KHash>
class makeJoin<7,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>(),
my_struct_key<K,T3>(),
my_struct_key<K,T4>(),
my_struct_key<K,T5>(),
my_struct_key<K,T6>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<7,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4)),
tag_func<T3>(T3(5)),
tag_func<T4>(T4(6)),
tag_func<T5>(T5(7)),
tag_func<T6>(T6(8))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};
#endif

#if MAX_TUPLE_TEST_SIZE >= 8
template<typename JType, typename K, typename KHash>
class makeJoin<8,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
typedef typename tbb::flow::tuple_element<7, TType>::type T7;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>(),
my_struct_key<K,T3>(),
my_struct_key<K,T4>(),
my_struct_key<K,T5>(),
my_struct_key<K,T6>(),
my_struct_key<K,T7>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<8,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
typedef typename tbb::flow::tuple_element<7, TType>::type T7;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4)),
tag_func<T3>(T3(5)),
tag_func<T4>(T4(6)),
tag_func<T5>(T5(7)),
tag_func<T6>(T6(8)),
tag_func<T7>(T7(9))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};
#endif

#if MAX_TUPLE_TEST_SIZE >= 9
template<typename JType, typename K, typename KHash>
class makeJoin<9,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
typedef typename tbb::flow::tuple_element<7, TType>::type T7;
typedef typename tbb::flow::tuple_element<8, TType>::type T8;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>(),
my_struct_key<K,T3>(),
my_struct_key<K,T4>(),
my_struct_key<K,T5>(),
my_struct_key<K,T6>(),
my_struct_key<K,T7>(),
my_struct_key<K,T8>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<9,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
typedef typename tbb::flow::tuple_element<7, TType>::type T7;
typedef typename tbb::flow::tuple_element<8, TType>::type T8;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4)),
tag_func<T3>(T3(5)),
tag_func<T4>(T4(6)),
tag_func<T5>(T5(7)),
tag_func<T6>(T6(8)),
tag_func<T7>(T7(9)),
tag_func<T8>(T8(10))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};
#endif

#if MAX_TUPLE_TEST_SIZE >= 10
template<typename JType, typename K, typename KHash>
class makeJoin<10,JType,tbb::flow::key_matching<K,KHash> > {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
typedef typename tbb::flow::tuple_element<7, TType>::type T7;
typedef typename tbb::flow::tuple_element<8, TType>::type T8;
typedef typename tbb::flow::tuple_element<9, TType>::type T9;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
my_struct_key<K,T0>(),
my_struct_key<K,T1>(),
my_struct_key<K,T2>(),
my_struct_key<K,T3>(),
my_struct_key<K,T4>(),
my_struct_key<K,T5>(),
my_struct_key<K,T6>(),
my_struct_key<K,T7>(),
my_struct_key<K,T8>(),
my_struct_key<K,T9>()
);
return temp;
}
static void destroy(JType *p) { delete p; }
};

template<typename JType>
class makeJoin<10,JType,tbb::flow::tag_matching> {
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple_element<0, TType>::type T0;
typedef typename tbb::flow::tuple_element<1, TType>::type T1;
typedef typename tbb::flow::tuple_element<2, TType>::type T2;
typedef typename tbb::flow::tuple_element<3, TType>::type T3;
typedef typename tbb::flow::tuple_element<4, TType>::type T4;
typedef typename tbb::flow::tuple_element<5, TType>::type T5;
typedef typename tbb::flow::tuple_element<6, TType>::type T6;
typedef typename tbb::flow::tuple_element<7, TType>::type T7;
typedef typename tbb::flow::tuple_element<8, TType>::type T8;
typedef typename tbb::flow::tuple_element<9, TType>::type T9;
public:
static JType *create(tbb::flow::graph& g) {
JType *temp = new JType(g,
tag_func<T0>(T0(2)),
tag_func<T1>(T1(3)),
tag_func<T2>(T2(4)),
tag_func<T3>(T3(5)),
tag_func<T4>(T4(6)),
tag_func<T5>(T5(7)),
tag_func<T6>(T6(8)),
tag_func<T7>(T7(9)),
tag_func<T8>(T8(10)),
tag_func<T9>(T9(11))
);
return temp;
}
static void destroy(JType *p) { delete p; }
};
#endif


static void* all_source_nodes[MaxPorts][MaxNSources];

template<int ELEM, typename JNT>
class source_node_helper {
public:
typedef JNT join_node_type;
typedef tbb::flow::join_node<tbb::flow::tuple<int, tbb::flow::continue_msg>, tbb::flow::reserving> input_join_type;
typedef typename join_node_type::output_type TT;
typedef typename tbb::flow::tuple_element<ELEM-1,TT>::type IT;
typedef typename tbb::flow::source_node<IT> my_source_node_type;
typedef typename tbb::flow::function_node<tbb::flow::tuple<int,tbb::flow::continue_msg>, IT> my_recirc_function_type;
static void print_remark(const char * str) {
source_node_helper<ELEM-1,JNT>::print_remark(str);
REMARK(", %s", name_of<IT>::name());
}
static void add_source_nodes(join_node_type &my_join, tbb::flow::graph &g, int nInputs) {
for(int i=0; i < nInputs; ++i) {
my_source_node_type *new_node = new my_source_node_type(g, source_body<IT,ELEM>(i, nInputs));
tbb::flow::make_edge( *new_node, tbb::flow::input_port<ELEM-1>(my_join) );
all_source_nodes[ELEM-1][i] = (void *)new_node;
}
source_node_helper<ELEM-1, JNT>::add_source_nodes(my_join, g, nInputs);
}

static void add_recirc_func_nodes(join_node_type &my_join, input_join_type &my_input, tbb::flow::graph &g) {
my_recirc_function_type *new_node = new my_recirc_function_type(g, tbb::flow::unlimited, recirc_func_body<IT>((IT)(ELEM+1)));
tbb::flow::make_edge(*new_node, tbb::flow::input_port<ELEM-1>(my_join));
tbb::flow::make_edge(my_input, *new_node);
all_source_nodes[ELEM-1][0] = (void *)new_node;
source_node_helper<ELEM-1, JNT>::add_recirc_func_nodes(my_join, my_input, g);
}

static void only_check_value(const int i, const TT &v) {
ASSERT( tbb::flow::get<ELEM-1>(v) == (IT)(i*(ELEM+1)), NULL);
source_node_helper<ELEM-1,JNT>::only_check_value(i, v);
}

static void check_value(int i, TT &v, bool is_serial) {
ASSERT(!is_serial || tbb::flow::get<ELEM-1>(v) == (IT)(i*(ELEM+1)), NULL);
int ival = (int)tbb::flow::get<ELEM-1>(v);
ASSERT(!(ival%(ELEM+1)), NULL);
ival /= (ELEM+1);
ASSERT(!outputCheck[ELEM-1][ival], NULL);
outputCheck[ELEM-1][ival] = true;
source_node_helper<ELEM-1,JNT>::check_value(i, v, is_serial);
}
static void remove_source_nodes(join_node_type& my_join, int nInputs) {
for(int i=0; i< nInputs; ++i) {
my_source_node_type *dp = reinterpret_cast<my_source_node_type *>(all_source_nodes[ELEM-1][i]);
tbb::flow::remove_edge( *dp, tbb::flow::input_port<ELEM-1>(my_join) );
delete dp;
}
source_node_helper<ELEM-1, JNT>::remove_source_nodes(my_join, nInputs);
}

static void remove_recirc_func_nodes(join_node_type& my_join, input_join_type &my_input) {
my_recirc_function_type *fn = reinterpret_cast<my_recirc_function_type *>(all_source_nodes[ELEM-1][0]);
tbb::flow::remove_edge( *fn, tbb::flow::input_port<ELEM-1>(my_join) );
tbb::flow::remove_edge( my_input, *fn);
delete fn;
source_node_helper<ELEM-1, JNT>::remove_recirc_func_nodes(my_join,my_input);
}
};

template<typename JNT>
class source_node_helper<1, JNT> {
typedef JNT join_node_type;
typedef tbb::flow::join_node<tbb::flow::tuple<int, tbb::flow::continue_msg>, tbb::flow::reserving> input_join_type;
typedef typename join_node_type::output_type TT;
typedef typename tbb::flow::tuple_element<0,TT>::type IT;
typedef typename tbb::flow::source_node<IT> my_source_node_type;
typedef typename tbb::flow::function_node<tbb::flow::tuple<int,tbb::flow::continue_msg>, IT> my_recirc_function_type;
public:
static void print_remark(const char * str) {
REMARK("%s< %s", str, name_of<IT>::name());
}
static void add_source_nodes(join_node_type &my_join, tbb::flow::graph &g, int nInputs) {
for(int i=0; i < nInputs; ++i) {
my_source_node_type *new_node = new my_source_node_type(g, source_body<IT,1>(i, nInputs));
tbb::flow::make_edge( *new_node, tbb::flow::input_port<0>(my_join) );
all_source_nodes[0][i] = (void *)new_node;
}
}

static void add_recirc_func_nodes(join_node_type &my_join, input_join_type &my_input, tbb::flow::graph &g) {
my_recirc_function_type *new_node = new my_recirc_function_type(g, tbb::flow::unlimited, recirc_func_body<IT>((IT)(2)));
tbb::flow::make_edge(*new_node, tbb::flow::input_port<0>(my_join));
tbb::flow::make_edge(my_input, *new_node);
all_source_nodes[0][0] = (void *)new_node;
}

static void only_check_value(const int i, const TT &v) {
ASSERT( tbb::flow::get<0>(v) == (IT)(i*2), NULL);
}

static void check_value(int i, TT &v, bool is_serial) {
ASSERT(!is_serial || tbb::flow::get<0>(v) == (IT)(i*(2)), NULL);
int ival = (int)tbb::flow::get<0>(v);
ASSERT(!(ival%2), NULL);
ival /= 2;
ASSERT(!outputCheck[0][ival], NULL);
outputCheck[0][ival] = true;
}
static void remove_source_nodes(join_node_type& my_join, int nInputs) {
for(int i=0; i < nInputs; ++i) {
my_source_node_type *dp = reinterpret_cast<my_source_node_type *>(all_source_nodes[0][i]);
tbb::flow::remove_edge( *dp, tbb::flow::input_port<0>(my_join) );
delete dp;
}
}

static void remove_recirc_func_nodes(join_node_type& my_join, input_join_type &my_input) {
my_recirc_function_type *fn = reinterpret_cast<my_recirc_function_type *>(all_source_nodes[0][0]);
tbb::flow::remove_edge( *fn, tbb::flow::input_port<0>(my_join) );
tbb::flow::remove_edge( my_input, *fn);
delete fn;
}
};

template<typename OutputTupleType>
class recirc_output_func_body {
public:
typedef typename tbb::flow::join_node<OutputTupleType, tbb::flow::tag_matching> join_node_type;
static const int N = tbb::flow::tuple_size<OutputTupleType>::value;
int operator()(const OutputTupleType &v) {
int out = int(tbb::flow::get<0>(v)) / 2;
source_node_helper<N,join_node_type>::only_check_value(out,v);
++output_count;
return out;
}
};

template<typename JType>
class tag_recirculation_test {
public:
typedef typename JType::output_type TType;
typedef typename tbb::flow::tuple<int, tbb::flow::continue_msg> input_tuple_type;
typedef tbb::flow::join_node<input_tuple_type,tbb::flow::reserving> input_join_type;
static const int N = tbb::flow::tuple_size<TType>::value;
static void test() {
source_node_helper<N,JType>::print_remark("Recirculation test of tag-matching join");
REMARK(" >\n");
for(int maxTag = 1; maxTag <10; maxTag *= 3) {
for(int i=0; i < N; ++i) all_source_nodes[i][0] = NULL;

tbb::flow::graph g;
JType * my_join = makeJoin<N,JType, tbb::flow::tag_matching>::create(g);
tbb::flow::source_node<tbb::flow::continue_msg> snode(g, recirc_source_node_body(), false);
input_join_type * my_input_join = makeJoin<2,input_join_type,tbb::flow::reserving>::create(g);
tbb::flow::make_edge(snode, tbb::flow::get<1>(my_input_join->input_ports()));
tbb::flow::queue_node<int> tag_queue(g);
tbb::flow::make_edge(tag_queue, tbb::flow::input_port<0>(*my_input_join));
source_node_helper<N,JType>::add_recirc_func_nodes(*my_join, *my_input_join, g);
tbb::flow::function_node<TType, int> recreate_tag(g, tbb::flow::unlimited, recirc_output_func_body<TType>());
tbb::flow::make_edge(*my_join, recreate_tag);
tbb::flow::make_edge(recreate_tag, tag_queue);

for(int t = 1; t <= maxTag; ++t) tag_queue.try_put(t);

input_count = Recirc_count;
output_count = 0;

snode.activate();

g.wait_for_all();

ASSERT(output_count == Recirc_count, "not all instances were received");

int j;
std::vector<bool> out_tally(maxTag, false);
for(int i = 0; i < maxTag; ++i) {
ASSERT(tag_queue.try_get(j), "not enough tags in queue");
ASSERT(!out_tally.at(j-1), "duplicate tag from queue");
out_tally[j-1] = true;
}
ASSERT(!tag_queue.try_get(j), "Extra tags in recirculation queue");

source_node_helper<N, JType>::remove_recirc_func_nodes(*my_join, *my_input_join);
tbb::flow::remove_edge(*my_join, recreate_tag);
makeJoin<N,JType,tbb::flow::tag_matching>::destroy(my_join);
tbb::flow::remove_edge(tag_queue, tbb::flow::input_port<0>(*my_input_join));
tbb::flow::remove_edge(snode, tbb::flow::input_port<1>(*my_input_join));
makeJoin<2,input_join_type,tbb::flow::reserving>::destroy(my_input_join);
}
}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif

template<typename JType, class JP>
class parallel_test {
public:
typedef typename JType::output_type TType;
static const int TUPLE_SIZE = tbb::flow::tuple_size<TType>::value;
typedef typename is_key_matching_join<JP>::key_type            key_type;
static const bool is_key_matching = is_key_matching_join<JP>::value;
static void test() {
TType v;
source_node_helper<TUPLE_SIZE,JType>::print_remark("Parallel test of join_node");
REMARK(" > ");
if(is_key_matching) {
REMARK("with K == %s", name_of<typename K_deref<typename is_key_matching_join<JP>::key_type>::type >::name());
if(is_ref<typename is_key_matching_join<JP>::key_type>::value) {
REMARK("&");
}
}
REMARK("\n");
for(int i=0; i < MaxPorts; ++i) {
for(int j=0; j < MaxNSources; ++j) {
all_source_nodes[i][j] = NULL;
}
}
for(int nInputs = 1; nInputs <= MaxNSources; ++nInputs) {
tbb::flow::graph g;
bool not_out_of_order = (nInputs == 1) && (!is_key_matching);
JType* my_join = makeJoin<TUPLE_SIZE,JType,JP>::create(g);
tbb::flow::queue_node<TType> outq1(g);
tbb::flow::queue_node<TType> outq2(g);

tbb::flow::make_edge( *my_join, outq1 );
tbb::flow::make_edge( *my_join, outq2 );

source_node_helper<TUPLE_SIZE, JType>::add_source_nodes((*my_join), g, nInputs);

g.wait_for_all();

reset_outputCheck(TUPLE_SIZE, Count);
for(int i=0; i < Count; ++i) {
ASSERT(outq1.try_get(v), NULL);
source_node_helper<TUPLE_SIZE, JType>::check_value(i, v, not_out_of_order);
}

check_outputCheck(TUPLE_SIZE, Count);
reset_outputCheck(TUPLE_SIZE, Count);

for(int i=0; i < Count; i++) {
ASSERT(outq2.try_get(v), NULL);;
source_node_helper<TUPLE_SIZE, JType>::check_value(i, v, not_out_of_order);
}
check_outputCheck(TUPLE_SIZE, Count);

ASSERT(!outq1.try_get(v), NULL);
ASSERT(!outq2.try_get(v), NULL);

source_node_helper<TUPLE_SIZE, JType>::remove_source_nodes((*my_join), nInputs);
tbb::flow::remove_edge( *my_join, outq1 );
tbb::flow::remove_edge( *my_join, outq2 );
makeJoin<TUPLE_SIZE,JType,JP>::destroy(my_join);
}
}
};


template<int ELEM, typename JType>
class serial_queue_helper {
public:
typedef typename JType::output_type TT;
typedef typename tbb::flow::tuple_element<ELEM-1,TT>::type IT;
typedef typename tbb::flow::queue_node<IT> my_queue_node_type;
static void print_remark() {
serial_queue_helper<ELEM-1,JType>::print_remark();
REMARK(", %s", name_of<IT>::name());
}
static void add_queue_nodes(tbb::flow::graph &g, JType &my_join) {
serial_queue_helper<ELEM-1,JType>::add_queue_nodes(g, my_join);
my_queue_node_type *new_node = new my_queue_node_type(g);
tbb::flow::make_edge( *new_node, tbb::flow::get<ELEM-1>(my_join.input_ports()) );
all_source_nodes[ELEM-1][0] = (void *)new_node;
}

static void fill_one_queue(int maxVal) {
my_queue_node_type *qptr = reinterpret_cast<my_queue_node_type *>(all_source_nodes[ELEM-1][0]);
serial_queue_helper<ELEM-1,JType>::fill_one_queue(maxVal);
for(int i = 0; i < maxVal; ++i) {
ASSERT(qptr->try_put(make_thingie<IT,ELEM>()(i)), NULL);
}
}

static void put_one_queue_val(int myVal) {
serial_queue_helper<ELEM-1,JType>::put_one_queue_val(myVal);
my_queue_node_type *qptr = reinterpret_cast<my_queue_node_type *>(all_source_nodes[ELEM-1][0]);
ASSERT(qptr->try_put(make_thingie<IT,ELEM>()(myVal)), NULL);
}

static void check_queue_value(int i, TT &v) {
serial_queue_helper<ELEM-1,JType>::check_queue_value(i, v);
ASSERT( cast_from<IT>::my_int_val(tbb::flow::get<ELEM-1>(v)) == i * (ELEM+1), NULL);
}

static void remove_queue_nodes(JType &my_join) {
my_queue_node_type *vptr = reinterpret_cast<my_queue_node_type *>(all_source_nodes[ELEM-1][0]);
tbb::flow::remove_edge( *vptr, tbb::flow::get<ELEM-1>(my_join.input_ports()) );
serial_queue_helper<ELEM-1, JType>::remove_queue_nodes(my_join);
delete vptr;
}
};

template<typename JType>
class serial_queue_helper<1, JType> {
public:
typedef typename JType::output_type TT;
typedef typename tbb::flow::tuple_element<0,TT>::type IT;
typedef typename tbb::flow::queue_node<IT> my_queue_node_type;
static void print_remark() {
REMARK("Serial test of join_node< %s", name_of<IT>::name());
}

static void add_queue_nodes(tbb::flow::graph &g, JType &my_join) {
my_queue_node_type *new_node = new my_queue_node_type(g);
tbb::flow::make_edge( *new_node, tbb::flow::input_port<0>(my_join) );
all_source_nodes[0][0] = (void *)new_node;
}

static void fill_one_queue(int maxVal) {
my_queue_node_type *qptr = reinterpret_cast<my_queue_node_type *>(all_source_nodes[0][0]);
for(int i = 0; i < maxVal; ++i) {
ASSERT(qptr->try_put(make_thingie<IT,1>()(i)), NULL);
}
}

static void put_one_queue_val(int myVal) {
my_queue_node_type *qptr = reinterpret_cast<my_queue_node_type *>(all_source_nodes[0][0]);
IT my_val = make_thingie<IT,1>()(myVal);
ASSERT(qptr->try_put(my_val), NULL);
}

static void check_queue_value(int i, TT &v) {
ASSERT( cast_from<IT>::my_int_val(tbb::flow::get<0>(v)) == i * 2, NULL);
}

static void remove_queue_nodes(JType &my_join) {
my_queue_node_type *vptr = reinterpret_cast<my_queue_node_type *>(all_source_nodes[0][0]);
tbb::flow::remove_edge( *vptr, tbb::flow::get<0>(my_join.input_ports()) );
delete vptr;
}
};


template<typename JType, class JP>
void test_one_serial( JType &my_join, tbb::flow::graph &g) {
typedef typename JType::output_type TType;
static const int TUPLE_SIZE = tbb::flow::tuple_size<TType>::value;
bool is_key_matching = is_key_matching_join<JP>::value;
std::vector<bool> flags;
serial_queue_helper<TUPLE_SIZE, JType>::add_queue_nodes(g,my_join);
typedef TType q3_input_type;
tbb::flow::queue_node< q3_input_type >  q3(g);

tbb::flow::make_edge( my_join, q3 );

flags.clear();
for (int i = 0; i < Count; ++i ) {
serial_queue_helper<TUPLE_SIZE,JType>::put_one_queue_val(i);
flags.push_back(false);
}

g.wait_for_all();
for (int i = 0; i < Count; ++i ) {
q3_input_type v;
g.wait_for_all();
ASSERT(q3.try_get( v ), "Error in try_get()");
if(is_key_matching) {
int j = int(tbb::flow::get<0>(v)) / 2;  
serial_queue_helper<TUPLE_SIZE,JType>::check_queue_value(j, v);
flags[j] = true;
}
else {
serial_queue_helper<TUPLE_SIZE,JType>::check_queue_value(i, v);
}
}

if(is_key_matching) {
for(int i = 0; i < Count; ++i) {
ASSERT(flags[i], NULL);
flags[i] = false;
}
}

serial_queue_helper<TUPLE_SIZE, JType>::fill_one_queue(Count);

g.wait_for_all();
for (int i = 0; i < Count; ++i ) {
q3_input_type v;
g.wait_for_all();
ASSERT(q3.try_get( v ), "Error in try_get()");
if(is_key_matching) {
int j = int(tbb::flow::get<0>(v)) / 2;
serial_queue_helper<TUPLE_SIZE,JType>::check_queue_value(j, v);
flags[i] = true;
}
else {
serial_queue_helper<TUPLE_SIZE,JType>::check_queue_value(i, v);
}
}

if(is_key_matching) {
for(int i = 0; i < Count; ++i) {
ASSERT(flags[i], NULL);
}
}

serial_queue_helper<TUPLE_SIZE, JType>::remove_queue_nodes(my_join);

}

template<typename JType, class JP>
class serial_test {
typedef typename JType::output_type TType;
static const int TUPLE_SIZE = tbb::flow::tuple_size<TType>::value;
static const int ELEMS = 3;
public:
static void test() {
tbb::flow::graph g;
std::vector<bool> flags;
bool is_key_matching = is_key_matching_join<JP>::value;
flags.reserve(Count);
JType* my_join = makeJoin<TUPLE_SIZE,JType,JP>::create(g);
serial_queue_helper<TUPLE_SIZE, JType>::print_remark(); REMARK(" >");
if(is_key_matching) {
REMARK("with K == %s", name_of<typename K_deref<typename is_key_matching_join<JP>::key_type>::type >::name());
if(is_ref<typename is_key_matching_join<JP>::key_type>::value) {
REMARK("&");
}
}
REMARK("\n");

test_one_serial<JType,JP>( *my_join, g);
std::vector<JType>join_vector(ELEMS, *my_join);
makeJoin<TUPLE_SIZE,JType,JP>::destroy(my_join);


for(int e = 0; e < ELEMS; ++e) {  
test_one_serial<JType,JP>( join_vector[e], g);
}
}

}; 

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif

template<
template<typename, class > class TestType,  
typename OutputTupleType,           
class J>                 
class generate_test {
public:
typedef tbb::flow::join_node<OutputTupleType,typename filter_out_message_based_key_matching<J>::policy> join_node_type;
static void do_test() {
TestType<join_node_type,J>::test();
}
};

template<typename JType>
class generate_recirc_test {
public:
typedef tbb::flow::join_node<JType, tbb::flow::tag_matching> join_node_type;
static void do_test() {
tag_recirculation_test<join_node_type>::test();
}
};

template<class JP>
void test_input_port_policies();

template<>
void test_input_port_policies<tbb::flow::reserving>() {
tbb::flow::graph g;
typedef tbb::flow::join_node<tbb::flow::tuple<int, int>, tbb::flow::reserving > JType; 
JType jn(g);
typedef JType::output_type OQType;
tbb::flow::queue_node<OQType> oq0(g);
tbb::flow::queue_node<OQType> oq1(g);
typedef tbb::flow::queue_node<int> IQType;
IQType iq0(g);
IQType iq1(g);
IQType qnp(g);
IQType qnq(g);
REMARK("Testing policies of join_node<reserving> input ports\n");
tbb::flow::make_edge( jn, oq0 );
tbb::flow::make_edge( jn, oq1 );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
ASSERT(jn.successor_count() == 2, NULL);
JType::successor_list_type my_succs;
jn.copy_successors(my_succs);
ASSERT(my_succs.size() == 2, NULL);
#endif
tbb::flow::make_edge( iq0, tbb::flow::get<0>(jn.input_ports()) );
tbb::flow::make_edge( iq1, tbb::flow::get<1>(jn.input_ports()) );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
ASSERT(tbb::flow::get<0>(jn.input_ports()).predecessor_count() == 1, NULL);
tbb::flow::tuple_element<0,JType::input_type>::type::predecessor_list_type my_0preds;
tbb::flow::input_port<0>(jn).copy_predecessors(my_0preds);
ASSERT(my_0preds.size() == 1, NULL);
#endif
for(int loop = 0; loop < 3; ++loop) {
ASSERT(iq0.try_put(1), "Error putting to iq1");
tbb::flow::make_edge( iq0, qnp );
g.wait_for_all();
{
int i;
ASSERT(qnp.try_get(i) && i == 1, "Error in item fetched by qnp");
}
ASSERT(iq1.try_put(2), "Error putting to iq1");
g.wait_for_all();
{
OQType t1;
ASSERT(!oq0.try_get(t1) && !oq1.try_get(t1), "oq0 and oq1 not empty");
}
tbb::flow::remove_edge( iq0, qnp); 
ASSERT(iq0.try_put(3), "Error on second put to iq0");
g.wait_for_all();
{
OQType t0;
OQType t1;
ASSERT(oq0.try_get(t0) && tbb::flow::get<0>(t0) == 3 && tbb::flow::get<1>(t0) == 2, "Error in oq0 output");
ASSERT(oq1.try_get(t1) && tbb::flow::get<0>(t1) == 3 && tbb::flow::get<1>(t1) == 2, "Error in oq1 output");
}
tbb::flow::make_edge( iq0, qnp );
tbb::flow::make_edge( iq1, qnq );
g.wait_for_all();
{
int i;
ASSERT(!qnp.try_get(i), "iq0 still had value in it");
ASSERT(!qnq.try_get(i), "iq1 still had value in it");
}
tbb::flow::remove_edge( iq0, qnp );
tbb::flow::remove_edge( iq1, qnq );
} 
}

template<>
void test_input_port_policies<tbb::flow::queueing>() {
tbb::flow::graph g;
typedef tbb::flow::join_node<tbb::flow::tuple<int, int>, tbb::flow::queueing > JType;
JType jn(g);
typedef JType::output_type OQType;
tbb::flow::queue_node<OQType> oq0(g);
tbb::flow::queue_node<OQType> oq1(g);
typedef tbb::flow::queue_node<int> IQType;
IQType iq0(g);
IQType iq1(g);
IQType qnp(g);
IQType qnq(g);
REMARK("Testing policies of join_node<queueing> input ports\n");
tbb::flow::make_edge( jn, oq0 );
tbb::flow::make_edge( jn, oq1 );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
ASSERT(jn.successor_count() == 2, NULL);
JType::successor_list_type my_succs;
jn.copy_successors(my_succs);
ASSERT(my_succs.size() == 2, NULL);
#endif
tbb::flow::make_edge( iq0, tbb::flow::get<0>(jn.input_ports()) );
tbb::flow::make_edge( iq1, tbb::flow::get<1>(jn.input_ports()) );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
ASSERT(tbb::flow::get<0>(jn.input_ports()).predecessor_count() == 1, NULL);
tbb::flow::tuple_element<0,JType::input_type>::type::predecessor_list_type my_0preds;
tbb::flow::input_port<0>(jn).copy_predecessors(my_0preds);
ASSERT(my_0preds.size() == 1, NULL);
#endif
for(int loop = 0; loop < 3; ++loop) {
ASSERT(iq0.try_put(1), "Error putting to iq1");
tbb::flow::make_edge( iq0, qnp );
g.wait_for_all();
{
int i;
ASSERT(!qnp.try_get(i), "Item was received by qnp");
}
ASSERT(iq1.try_put(2), "Error putting to iq1");
g.wait_for_all();
{
OQType t0;
OQType t1;
ASSERT(oq0.try_get(t0) && tbb::flow::get<0>(t0) == 1 && tbb::flow::get<1>(t0) == 2, "Error in oq0 output");
ASSERT(oq1.try_get(t1) && tbb::flow::get<0>(t1) == 1 && tbb::flow::get<1>(t1) == 2, "Error in oq1 output");
}
tbb::flow::make_edge( iq1, qnq );
g.wait_for_all();
{
int i;
ASSERT(!qnp.try_get(i), "iq0 still had value in it");
ASSERT(!qnq.try_get(i), "iq1 still had value in it");
}
tbb::flow::remove_edge( iq0, qnp );
tbb::flow::remove_edge( iq1, qnq );
} 
}

template<typename T>
struct myTagValue {
tbb::flow::tag_value operator()(T i) { return tbb::flow::tag_value(i); }
};

template<>
struct myTagValue<check_type<int> > {
tbb::flow::tag_value operator()(check_type<int> i) { return tbb::flow::tag_value((int)i); }
};

template<>
void test_input_port_policies<tbb::flow::tag_matching>() {
tbb::flow::graph g;
typedef tbb::flow::join_node<tbb::flow::tuple<int, check_type<int> >, tbb::flow::tag_matching > JoinNodeType;
typedef JoinNodeType::output_type CheckTupleType;
JoinNodeType testJoinNode(g, myTagValue<int>(), myTagValue<check_type<int> >());
tbb::flow::queue_node<CheckTupleType> checkTupleQueue0(g);
tbb::flow::queue_node<CheckTupleType> checkTupleQueue1(g);
{
Check<check_type<int> > my_check;


typedef tbb::flow::queue_node<int> IntQueueType;
typedef tbb::flow::queue_node<check_type<int> > CheckQueueType;
IntQueueType intInputQueue(g);
CheckQueueType checkInputQueue(g);
IntQueueType intEmptyTestQueue(g);
CheckQueueType checkEmptyTestQueue(g);
REMARK("Testing policies of join_node<tag_matching> input ports\n");
tbb::flow::make_edge( testJoinNode, checkTupleQueue0 );
tbb::flow::make_edge( testJoinNode, checkTupleQueue1 );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
ASSERT(testJoinNode.successor_count() == 2, NULL);
JoinNodeType::successor_list_type my_succs;
testJoinNode.copy_successors(my_succs);
ASSERT(my_succs.size() == 2, NULL);
#endif
tbb::flow::make_edge( intInputQueue, tbb::flow::input_port<0>(testJoinNode) );
tbb::flow::make_edge( checkInputQueue, tbb::flow::input_port<1>(testJoinNode) );
#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
ASSERT(tbb::flow::input_port<0>(testJoinNode).predecessor_count() == 1, NULL);
tbb::flow::tuple_element<0,JoinNodeType::input_type>::type::predecessor_list_type my_0preds;
tbb::flow::input_port<0>(testJoinNode).copy_predecessors(my_0preds);
ASSERT(my_0preds.size() == 1, NULL);
#endif

for(int loop = 0; loop < 4; ++loop) {
ASSERT(intInputQueue.try_put(loop), "Error putting to intInputQueue");
tbb::flow::make_edge( intInputQueue, intEmptyTestQueue );
g.wait_for_all();
{
int intVal0;
ASSERT(!intEmptyTestQueue.try_get(intVal0), "Item was received by intEmptyTestQueue");
}
check_type<int> checkVal0(loop);
ASSERT(checkInputQueue.try_put(checkVal0), "Error putting to checkInputQueue");
g.wait_for_all();
{
CheckTupleType t0;
CheckTupleType t1;
ASSERT(checkTupleQueue0.try_get(t0) && tbb::flow::get<0>(t0) == loop && (int)tbb::flow::get<1>(t0) == loop, "Error in checkTupleQueue0 output");
ASSERT(checkTupleQueue1.try_get(t1) && tbb::flow::get<0>(t1) == loop && (int)tbb::flow::get<1>(t1) == loop, "Error in checkTupleQueue1 output");
ASSERT(!checkTupleQueue0.try_get(t0), "extra object in output queue checkTupleQueue0");
ASSERT(!checkTupleQueue1.try_get(t0), "extra object in output queue checkTupleQueue1");
}
tbb::flow::make_edge( checkInputQueue, checkEmptyTestQueue );
g.wait_for_all();
{
int intVal1;
check_type<int> checkVal1;
ASSERT(!intEmptyTestQueue.try_get(intVal1), "intInputQueue still had value in it");
ASSERT(!checkEmptyTestQueue.try_get(checkVal1), "checkInputQueue still had value in it");
}
tbb::flow::remove_edge( intInputQueue, intEmptyTestQueue );
tbb::flow::remove_edge( checkInputQueue, checkEmptyTestQueue );
} 

const int nValues = 100;
const int nIncr = 31;  

for(int loop = 4; loop < 4+nValues; ++loop) {
ASSERT(intInputQueue.try_put(loop), "Error putting to intInputQueue");
g.wait_for_all();
{
CheckTupleType t3;
ASSERT(!checkTupleQueue0.try_get(t3), "Object in output queue");
ASSERT(!checkTupleQueue1.try_get(t3), "Object in output queue");
}
} 

for(int loop = 1; loop <= nValues; ++loop) {
int lp1 = 4 + (loop * nIncr)%nValues;
ASSERT(checkInputQueue.try_put(lp1), "Error putting to checkInputQueue");
g.wait_for_all();
{
CheckTupleType t0;
CheckTupleType t1;
ASSERT(checkTupleQueue0.try_get(t0) && tbb::flow::get<0>(t0) == lp1 && tbb::flow::get<1>(t0) == lp1, "Error in checkTupleQueue0 output");
ASSERT(checkTupleQueue1.try_get(t1) && tbb::flow::get<0>(t1) == lp1 && tbb::flow::get<1>(t1) == lp1, "Error in checkTupleQueue1 output");
ASSERT(!checkTupleQueue0.try_get(t0), "extra object in output queue checkTupleQueue0");
ASSERT(!checkTupleQueue1.try_get(t0), "extra object in output queue checkTupleQueue1");
}
} 
} 
}

int TestMain() {
#if __TBB_USE_TBB_TUPLE
REMARK("  Using TBB tuple\n");
#else
REMARK("  Using platform tuple\n");
#endif
TestTaggedBuffers();
test_input_port_policies<tbb::flow::reserving>();
test_input_port_policies<tbb::flow::queueing>();
test_input_port_policies<tbb::flow::tag_matching>();
for (int p = 0; p < 2; ++p) {
REMARK("reserving\n");
generate_test<serial_test, tbb::flow::tuple<threebyte, double>, tbb::flow::reserving >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 4
{
Check<check_type<int> > my_check;
generate_test<serial_test, tbb::flow::tuple<float, double, check_type<int>, long>, tbb::flow::reserving >::do_test();
}
#endif
#if MAX_TUPLE_TEST_SIZE >= 6
generate_test<serial_test, tbb::flow::tuple<double, double, int, long, int, short>, tbb::flow::reserving >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 8
generate_test<serial_test, tbb::flow::tuple<float, double, double, double, float, int, float, long>, tbb::flow::reserving >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 10
generate_test<serial_test, tbb::flow::tuple<float, double, int, double, double, float, long, int, float, long>, tbb::flow::reserving >::do_test();
#endif
{
Check<check_type<int> > my_check1;
generate_test<parallel_test, tbb::flow::tuple<float, check_type<int> >, tbb::flow::reserving >::do_test();
}
#if MAX_TUPLE_TEST_SIZE >= 3
generate_test<parallel_test, tbb::flow::tuple<float, int, long>, tbb::flow::reserving >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 5
generate_test<parallel_test, tbb::flow::tuple<double, double, int, int, short>, tbb::flow::reserving >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
generate_test<parallel_test, tbb::flow::tuple<float, int, double, float, long, float, long>, tbb::flow::reserving >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 9
generate_test<parallel_test, tbb::flow::tuple<float, double, int, double, double, long, int, float, long>, tbb::flow::reserving >::do_test();
#endif
REMARK("queueing\n");
generate_test<serial_test, tbb::flow::tuple<float, double>, tbb::flow::queueing >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 4
generate_test<serial_test, tbb::flow::tuple<float, double, int, long>, tbb::flow::queueing >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 6
generate_test<serial_test, tbb::flow::tuple<double, double, int, long, int, short>, tbb::flow::queueing >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 8
generate_test<serial_test, tbb::flow::tuple<float, double, double, double, float, int, float, long>, tbb::flow::queueing >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 10
generate_test<serial_test, tbb::flow::tuple<float, double, int, double, double, float, long, int, float, long>, tbb::flow::queueing >::do_test();
#endif
generate_test<parallel_test, tbb::flow::tuple<float, double>, tbb::flow::queueing >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 3
generate_test<parallel_test, tbb::flow::tuple<float, int, long>, tbb::flow::queueing >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 5
generate_test<parallel_test, tbb::flow::tuple<double, double, int, int, short>, tbb::flow::queueing >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
generate_test<parallel_test, tbb::flow::tuple<float, int, double, float, long, float, long>, tbb::flow::queueing >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 9
generate_test<parallel_test, tbb::flow::tuple<float, double, int, double, double, long, int, float, long>, tbb::flow::queueing >::do_test();
#endif
REMARK("tag_matching\n");
generate_test<serial_test, tbb::flow::tuple<float, double>, tbb::flow::tag_matching >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 4
generate_test<serial_test, tbb::flow::tuple<float, double, int, long>, tbb::flow::tag_matching >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 6
generate_test<serial_test, tbb::flow::tuple<double, double, int, long, int, short>, tbb::flow::tag_matching >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 8
generate_test<serial_test, tbb::flow::tuple<float, double, double, double, float, int, float, long>, tbb::flow::tag_matching >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 10
generate_test<serial_test, tbb::flow::tuple<float, double, int, double, double, float, long, int, float, long>, tbb::flow::tag_matching >::do_test();
#endif
generate_test<parallel_test, tbb::flow::tuple<float, double>, tbb::flow::tag_matching >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 3
generate_test<parallel_test, tbb::flow::tuple<float, int, long>, tbb::flow::tag_matching >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 5
generate_test<parallel_test, tbb::flow::tuple<double, double, int, int, short>, tbb::flow::tag_matching >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
generate_test<parallel_test, tbb::flow::tuple<float, int, double, float, long, float, long>, tbb::flow::tag_matching >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 9
generate_test<parallel_test, tbb::flow::tuple<float, double, int, double, double, long, int, float, long>, tbb::flow::tag_matching >::do_test();
#endif
generate_recirc_test<tbb::flow::tuple<float,double> >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 5
generate_recirc_test<tbb::flow::tuple<double, double, int, int, short> >::do_test();
#endif
}
REMARK("key_matching\n");
generate_test<serial_test,tbb::flow::tuple<MyKeyFirst<int,double>,MyKeySecond<int,float> >,tbb::flow::key_matching<int> >::do_test();
generate_test<serial_test,tbb::flow::tuple<MyKeyFirst<std::string,double>,MyKeySecond<std::string,float> >,tbb::flow::key_matching<std::string> >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 3
generate_test<serial_test,tbb::flow::tuple<MyKeyFirst<std::string,double>,MyKeySecond<std::string,float>, MyKeyWithBrokenMessageKey<std::string, int> >,tbb::flow::key_matching<std::string&> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
generate_test<serial_test,tbb::flow::tuple<
MyKeyFirst<std::string,double>,
MyKeyWithBrokenMessageKey<std::string,int>,
MyKeyFirst<std::string, int>,
MyKeySecond<std::string,size_t>,
MyKeyWithBrokenMessageKey<std::string, int>,
MyKeySecond<std::string,short>,
MyKeySecond<std::string,threebyte>
> ,tbb::flow::key_matching<std::string&> >::do_test();
#endif

generate_test<parallel_test,tbb::flow::tuple<MyKeyFirst<int,double>,MyKeySecond<int,float> >,tbb::flow::key_matching<int> >::do_test();
generate_test<parallel_test,tbb::flow::tuple<MyKeyFirst<int,double>,MyKeySecond<int,float> >,tbb::flow::key_matching<int&> >::do_test();
generate_test<parallel_test,tbb::flow::tuple<MyKeyFirst<std::string,double>,MyKeySecond<std::string,float> >,tbb::flow::key_matching<std::string&> >::do_test();
#if MAX_TUPLE_TEST_SIZE >= 10
generate_test<parallel_test,tbb::flow::tuple<
MyKeyFirst<std::string,double>,
MyKeySecond<std::string,int>,
MyKeyFirst<std::string, int>,
MyKeyWithBrokenMessageKey<std::string,size_t>,
MyKeyWithBrokenMessageKey<std::string, int>,
MyKeySecond<std::string,short>,
MyKeySecond<std::string,threebyte>,
MyKeyFirst<std::string, int>,
MyKeySecond<std::string,threebyte>,
MyKeyWithBrokenMessageKey<std::string,size_t>
> ,tbb::flow::key_matching<std::string&> >::do_test();
#endif

REMARK("message based key_matching\n");
generate_test<serial_test,tbb::flow::tuple<MyMessageKeyWithBrokenKey<int,double>,MyMessageKeyWithoutKey<int,float> >,message_based_key_matching<int> >::do_test();
generate_test<serial_test,tbb::flow::tuple<MyMessageKeyWithoutKeyMethod<std::string,double>,MyMessageKeyWithBrokenKey<std::string,float> >,message_based_key_matching<std::string> >::do_test();
#if !__TBB_MIC_OFFLOAD_TEST_COMPILATION_BROKEN
#if MAX_TUPLE_TEST_SIZE >= 3
generate_test<serial_test,tbb::flow::tuple<MyMessageKeyWithoutKey<std::string,double>,MyMessageKeyWithoutKeyMethod<std::string,float>,MyMessageKeyWithBrokenKey<std::string, int> >,message_based_key_matching<std::string&> >::do_test();
#endif
#if MAX_TUPLE_TEST_SIZE >= 7
generate_test<serial_test,tbb::flow::tuple<
MyMessageKeyWithoutKey<std::string,double>,
MyMessageKeyWithoutKeyMethod<std::string,int>,
MyMessageKeyWithBrokenKey<std::string, int>,
MyMessageKeyWithoutKey<std::string,size_t>,
MyMessageKeyWithoutKeyMethod<std::string, int>,
MyMessageKeyWithBrokenKey<std::string,short>,
MyMessageKeyWithoutKey<std::string,threebyte>
> ,message_based_key_matching<std::string&> >::do_test();
#endif

generate_test<parallel_test,tbb::flow::tuple<MyMessageKeyWithBrokenKey<int,double>,MyMessageKeyWithoutKey<int,float> >,message_based_key_matching<int> >::do_test();
generate_test<parallel_test,tbb::flow::tuple<MyMessageKeyWithoutKeyMethod<int,double>,MyMessageKeyWithBrokenKey<int,float> >,message_based_key_matching<int&> >::do_test();
generate_test<parallel_test,tbb::flow::tuple<MyMessageKeyWithoutKey<std::string,double>,MyMessageKeyWithoutKeyMethod<std::string,float> >,message_based_key_matching<std::string&> >::do_test();

#if MAX_TUPLE_TEST_SIZE >= 10
generate_test<parallel_test,tbb::flow::tuple<
MyMessageKeyWithoutKeyMethod<std::string,double>,
MyMessageKeyWithBrokenKey<std::string,int>,
MyMessageKeyWithoutKey<std::string, int>,
MyMessageKeyWithoutKeyMethod<std::string,size_t>,
MyMessageKeyWithBrokenKey<std::string, int>,
MyMessageKeyWithoutKeyMethod<std::string,short>,
MyMessageKeyWithoutKeyMethod<std::string,threebyte>,
MyMessageKeyWithBrokenKey<std::string, int>,
MyMessageKeyWithoutKeyMethod<std::string,threebyte>,
MyMessageKeyWithBrokenKey<std::string,size_t>
> ,message_based_key_matching<std::string&> >::do_test();
#endif
#endif 

#if TBB_PREVIEW_FLOW_GRAPH_FEATURES
REMARK("test queueing extract\n");
test_join_extract<int, tbb::flow::join_node< tbb::flow::tuple<int,int>, tbb::flow::queueing> >().run_tests();
REMARK("test tag_matching extract\n");
test_join_extract<int, tbb::flow::join_node< tbb::flow::tuple<int,int>, tbb::flow::tag_matching> >().run_tests();
REMARK("test reserving extract\n");
test_join_extract<int, tbb::flow::join_node< tbb::flow::tuple<int,int>, tbb::flow::reserving> >().run_tests();
#endif
return Harness::Done;
}
