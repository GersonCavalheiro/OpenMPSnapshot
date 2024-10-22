

#include "tbb/parallel_do.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/atomic.h"
#include "harness.h"
#include "harness_cpu.h"
#include <deque>

#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (disable: 4267)
#endif 

#define N_DEPTHS     20

static tbb::atomic<int> g_values_counter;

class value_t {
size_t x;
value_t& operator= ( const value_t& );
public:
value_t ( size_t xx ) : x(xx) { ++g_values_counter; }
value_t ( const value_t& v ) : x(v.x) { ++g_values_counter; }
#if __TBB_CPP11_RVALUE_REF_PRESENT
value_t ( value_t&& v ) : x(v.x) { ++g_values_counter; }
#endif
~value_t () { --g_values_counter; }
size_t value() const volatile { return x; }
};

#include "harness_iterator.h"

static size_t g_tasks_expected = 0;
static tbb::atomic<size_t> g_tasks_observed;

size_t FindNumOfTasks ( size_t max_depth ) {
if( max_depth == 0 )
return 1;
return  max_depth * FindNumOfTasks( max_depth - 1 ) + 1;
}

class FakeTaskGeneratorBody {
public:

void operator() ( value_t depth ) const {
g_tasks_observed += FindNumOfTasks(depth.value());
}
};


class FakeTaskGeneratorBody_RefVersion {
public:
void operator() ( value_t& depth ) const {
g_tasks_observed += FindNumOfTasks(depth.value());
}
};


class FakeTaskGeneratorBody_ConstRefVersion {
public:
void operator() ( const value_t& depth ) const {
g_tasks_observed += FindNumOfTasks(depth.value());
}
};


class FakeTaskGeneratorBody_VolatileRefVersion {
public:
void operator() ( volatile value_t& depth, tbb::parallel_do_feeder<value_t>& ) const {
g_tasks_observed += FindNumOfTasks(depth.value());
}
};

#if __TBB_CPP11_RVALUE_REF_PRESENT

class FakeTaskGeneratorBody_RvalueRefVersion {
public:
void operator() ( value_t&& depth ) const {
g_tasks_observed += FindNumOfTasks(depth.value());
}
};
#endif

void do_work ( const value_t& depth, tbb::parallel_do_feeder<value_t>& feeder ) {
++g_tasks_observed;
value_t new_value(depth.value()-1);
for( size_t i = 0; i < depth.value(); ++i) {
if (i%2) feeder.add( new_value ); 
else     feeder.add( value_t(depth.value()-1) ); 
}
}


class TaskGeneratorBody
{
public:
void operator() ( value_t depth, tbb::parallel_do_feeder<value_t>& feeder ) const {
do_work(depth, feeder);
}
private:
TaskGeneratorBody () {}
TaskGeneratorBody ( const TaskGeneratorBody& );
template<class Body, class Iterator> friend void TestBody( size_t );
template<class Body, class Iterator> friend void TestBody_MoveOnly( size_t );
};


class TaskGeneratorBody_RefVersion
{
public:
void operator() ( value_t& depth, tbb::parallel_do_feeder<value_t>& feeder ) const {
do_work(depth, feeder);
}
};


class TaskGeneratorBody_ConstVersion
{
public:
void operator() ( const value_t depth, tbb::parallel_do_feeder<value_t>& feeder ) const {
do_work(depth, feeder);
}
};


class TaskGeneratorBody_ConstRefVersion
{
public:
void operator() ( const value_t& depth, tbb::parallel_do_feeder<value_t>& feeder ) const {
do_work(depth, feeder);
}
};


class TaskGeneratorBody_VolatileRefVersion
{
public:
void operator() ( volatile value_t& depth, tbb::parallel_do_feeder<value_t>& feeder ) const {
do_work(const_cast<value_t&>(depth), feeder);
}
};


class TaskGeneratorBody_ConstVolatileRefVersion
{
public:
void operator() ( const volatile value_t& depth, tbb::parallel_do_feeder<value_t>& feeder ) const {
do_work(const_cast<value_t&>(depth), feeder);
}
};

#if __TBB_CPP11_RVALUE_REF_PRESENT

class TaskGeneratorBody_RvalueRefVersion
{
public:
void operator() ( value_t&& depth, tbb::parallel_do_feeder<value_t>& feeder ) const {
do_work(depth, feeder);
}
};
#endif

static value_t g_depths[N_DEPTHS] = {0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};

#if __TBB_CPP11_RVALUE_REF_PRESENT
template<class Body, class Iterator>
void TestBody_MoveIter ( const Body& body, Iterator begin, Iterator end  ) {
typedef std::move_iterator<Iterator> MoveIterator;
MoveIterator mbegin(begin);
MoveIterator mend(end);
g_tasks_observed = 0;
tbb::parallel_do(mbegin, mend, body);
ASSERT (g_tasks_observed == g_tasks_expected, NULL);
}

template<class Body, class Iterator>
void TestBody_MoveOnly ( size_t depth ) {
typedef typename std::iterator_traits<Iterator>::value_type value_type;
value_type a_depths[N_DEPTHS] = {0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
TestBody_MoveIter( Body(), Iterator(a_depths), Iterator(a_depths + depth));
}
#endif

template<class Body, class Iterator>
void TestBody ( size_t depth ) {
typedef typename std::iterator_traits<Iterator>::value_type value_type;
value_type a_depths[N_DEPTHS] = {0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
Body body;
Iterator begin(a_depths);
Iterator end(a_depths + depth);
g_tasks_observed = 0;
tbb::parallel_do(begin, end, body);
ASSERT (g_tasks_observed == g_tasks_expected, NULL);
#if __TBB_CPP11_RVALUE_REF_PRESENT
TestBody_MoveIter( body, Iterator(a_depths), Iterator(a_depths + depth) );
#endif
}

template<class Iterator>
void TestIterator_Common ( size_t depth ) {
TestBody<FakeTaskGeneratorBody, Iterator> (depth);
TestBody<FakeTaskGeneratorBody_ConstRefVersion, Iterator> (depth);
TestBody<TaskGeneratorBody, Iterator> (depth);
TestBody<TaskGeneratorBody_ConstVersion, Iterator> (depth);
TestBody<TaskGeneratorBody_ConstRefVersion, Iterator> (depth);
}

template<class Iterator>
void TestIterator_Const ( size_t depth ) {
TestIterator_Common<Iterator>(depth);
TestBody<TaskGeneratorBody_ConstVolatileRefVersion, Iterator> (depth);
}

template<class Iterator>
void TestIterator_Modifiable ( size_t depth ) {
TestIterator_Const<Iterator>(depth);
TestBody<FakeTaskGeneratorBody_RefVersion, Iterator> (depth);
TestBody<FakeTaskGeneratorBody_VolatileRefVersion, Iterator> (depth);
TestBody<TaskGeneratorBody_RefVersion, Iterator> (depth);
TestBody<TaskGeneratorBody_VolatileRefVersion, Iterator> (depth);
#if __TBB_CPP11_RVALUE_REF_PRESENT
TestBody_MoveOnly<FakeTaskGeneratorBody_RvalueRefVersion, Iterator> (depth);
TestBody_MoveOnly<TaskGeneratorBody_RvalueRefVersion, Iterator> (depth);
#endif
}

template<class Iterator>
void TestIterator_Movable ( size_t depth ) {
TestIterator_Common<Iterator>(depth);
#if __TBB_CPP11_RVALUE_REF_PRESENT
TestBody<FakeTaskGeneratorBody_RvalueRefVersion, Iterator> (depth);
TestBody<TaskGeneratorBody_RvalueRefVersion, Iterator> (depth);
#endif
}

void Run( int  ) {
for( size_t depth = 0; depth <= N_DEPTHS; ++depth ) {
g_tasks_expected = 0;
for ( size_t i=0; i < depth; ++i )
g_tasks_expected += FindNumOfTasks( g_depths[i].value() );
TestIterator_Movable<size_t*>(depth);
TestIterator_Modifiable<value_t*>(depth);
TestIterator_Modifiable<Harness::InputIterator<value_t> >(depth);
TestIterator_Modifiable<Harness::ForwardIterator<value_t> >(depth);
TestIterator_Const<Harness::ConstRandomIterator<value_t> >(depth);
}
}

const size_t elements = 10000;
const size_t init_sum = 0;
tbb::atomic<size_t> element_counter;

template<size_t K>
struct set_to {
void operator()(size_t& x) const {
x = K;
++element_counter;
}
};

#include "test_range_based_for.h"
#include <functional>

void range_do_test() {
using namespace range_based_for_support_tests;
std::deque<size_t> v(elements, 0);

element_counter = 0;
tbb::parallel_do(v.begin(), v.end(), set_to<1>());
ASSERT(element_counter == v.size() && element_counter == elements, "not all elements were set");
ASSERT(range_based_for_accumulate(v, std::plus<size_t>(), init_sum) == v.size(), "elements of v not all ones");

element_counter = 0;
tbb::parallel_do(v, set_to<0>());
ASSERT(element_counter == v.size() && element_counter == elements, "not all elements were set");
ASSERT(range_based_for_accumulate(v, std::plus<size_t>(), init_sum) == init_sum, "elements of v not all zeros");

element_counter = 0;
tbb::parallel_do(tbb::blocked_range<std::deque<size_t>::iterator>(v.begin(), v.end()), set_to<1>());
ASSERT(element_counter == v.size() && element_counter == elements, "not all elements were set");
ASSERT(range_based_for_accumulate(v, std::plus<size_t>(), init_sum) == v.size(), "elements of v not all ones");

element_counter = 0;
tbb::task_group_context context;
tbb::parallel_do(v.begin(), v.end(), set_to<0>(), context);
ASSERT(element_counter == v.size() && element_counter == elements, "not all elements were set");
ASSERT(range_based_for_accumulate(v, std::plus<size_t>(), init_sum) == init_sum, "elements of v not all ones");

element_counter = 0;
tbb::parallel_do(v, set_to<1>(), context);
ASSERT(element_counter == v.size() && element_counter == elements, "not all elements were set");
ASSERT(range_based_for_accumulate(v, std::plus<size_t>(), init_sum) == v.size(), "elements of v not all ones");

element_counter = 0;
tbb::parallel_do(tbb::blocked_range<std::deque<size_t>::iterator>(v.begin(), v.end()), set_to<0>(), context);
ASSERT(element_counter == v.size() && element_counter == elements, "not all elements were set");
ASSERT(range_based_for_accumulate(v, std::plus<size_t>(), init_sum) == init_sum, "elements of v not all zeros");
}

#if __TBB_CPP11_RVALUE_REF_PRESENT
struct Movable {
Movable() : alive(true), addtofeed(true) {}
Movable(bool addtofeed_) : alive(true), addtofeed(addtofeed_) {}
Movable(Movable&& other) {
ASSERT(other.alive, "Moving from a dead object");
alive = true;
addtofeed = other.addtofeed;
other.alive = false;
}
Movable& operator=(Movable&& other) {
ASSERT(alive, "Assignment to a dead object");
ASSERT(other.alive, "Assignment of a dead object");
other.alive = false;
return *this;
}
Movable(const Movable&) { REPORT("Error: copy ctor prefered.\n"); }
Movable& operator=(const Movable&) { REPORT("Error: copy assing operator prefered.\n"); return *this; }
~Movable() { alive = false; }
volatile bool alive;
bool addtofeed;
};

struct MoveOnly : Movable, NoCopy {
MoveOnly() : Movable() {}
MoveOnly(bool addtofeed_) : Movable(addtofeed_) {}
MoveOnly(MoveOnly&& other): Movable(std::move(other)) {}
};

template<typename T>
void RecordAndAdd(const T& in, tbb::parallel_do_feeder<T>& feeder) {
ASSERT(in.alive, "Got dead object in body");
size_t i = ++g_tasks_observed;
if (in.addtofeed) {
if (i%2) feeder.add(T(false));
else {
T a(false);
feeder.add(std::move(a));
}
}
}
template<typename T>
struct TestMoveIteratorBody {
void operator() (T&& in, tbb::parallel_do_feeder<T>& feeder) const { RecordAndAdd(in, feeder); }
};
template<typename T>
struct TestMoveIteratorBodyByValue {
void operator() (T in, tbb::parallel_do_feeder<T>& feeder) const { RecordAndAdd(in, feeder); }
};

template<typename Iterator, typename Body>
void TestMoveIterator() {
typedef typename std::iterator_traits<Iterator>::value_type value_type;

Body body;
const size_t size = 65;
g_tasks_observed = 0;
value_type a[size];
tbb::parallel_do( std::make_move_iterator(Iterator(a)), std::make_move_iterator(Iterator(a+size)), body );
ASSERT(size * 2  == g_tasks_observed, NULL);
}

template<typename T>
void DoTestMoveSemantics() {
TestMoveIterator< Harness::InputIterator<T>, TestMoveIteratorBody<T> >();
TestMoveIterator< Harness::ForwardIterator<T>, TestMoveIteratorBody<T> >();
TestMoveIterator< Harness::RandomIterator<T>, TestMoveIteratorBody<T> >();

TestMoveIterator< Harness::InputIterator<T>, TestMoveIteratorBodyByValue<T> >();
TestMoveIterator< Harness::ForwardIterator<T>, TestMoveIteratorBodyByValue<T> >();
TestMoveIterator< Harness::RandomIterator<T>, TestMoveIteratorBodyByValue<T> >();
}

void TestMoveSemantics() {
DoTestMoveSemantics<Movable>();
#if __TBB_CPP11_IS_COPY_CONSTRUCTIBLE_PRESENT && !__TBB_IS_COPY_CONSTRUCTIBLE_BROKEN
DoTestMoveSemantics<MoveOnly>();
#endif
}
#else 
void TestMoveSemantics() {
REPORT("Known issue: move support tests are skipped.\n");
}
#endif

int TestMain () {
if( MinThread<1 ) {
REPORT("number of threads must be positive\n");
exit(1);
}
g_values_counter = 0;
for( int p=MinThread; p<=MaxThread; ++p ) {
tbb::task_scheduler_init init( p );
Run(p);
range_do_test();
TestCPUUserTime(p);
}
ASSERT( g_values_counter == 0, "Value objects were leaked" );

TestMoveSemantics();
return Harness::Done;
}
