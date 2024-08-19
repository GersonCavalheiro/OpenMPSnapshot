

#define __TBB_EXTRA_DEBUG 1

#include <stdexcept>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <set>

#include "harness_fp.h"

#if __TBB_TASK_ISOLATION
#define private public
#include "tbb/task.h"
#undef private
#endif 

#include "tbb/task_arena.h"
#include "tbb/atomic.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/enumerable_thread_specific.h"

#include "harness_assert.h"
#include "harness.h"
#include "harness_barrier.h"

#include "tbb/tbb_thread.h"

#if _MSC_VER
#pragma comment(lib, __TBB_STRING(__TBB_LIB_NAME))
#endif

#include "tbb/global_control.h"

void InitializeAndTerminate( int maxthread ) {
__TBB_TRY {
for( int i=0; i<200; ++i ) {
switch( i&3 ) {
default: {
tbb::task_arena arena( std::rand() % maxthread + 1 );
ASSERT(!arena.is_active(), "arena should not be active until initialized");
arena.initialize();
ASSERT(arena.is_active(), NULL);
arena.terminate();
ASSERT(!arena.is_active(), "arena should not be active; it was terminated");
break;
}
case 0: {
tbb::task_arena arena( 1 );
ASSERT(!arena.is_active(), "arena should not be active until initialized");
arena.initialize( std::rand() % maxthread + 1 ); 
ASSERT(arena.is_active(), NULL);
break;
}
case 1: {
tbb::task_arena arena( tbb::task_arena::automatic );
ASSERT(!arena.is_active(), NULL);
arena.initialize();
ASSERT(arena.is_active(), NULL);
break;
}
case 2: {
tbb::task_arena arena;
ASSERT(!arena.is_active(), "arena should not be active until initialized");
arena.initialize( std::rand() % maxthread + 1 );
ASSERT(arena.is_active(), NULL);
arena.terminate();
ASSERT(!arena.is_active(), "arena should not be active; it was terminated");
break;
}
}
}
} __TBB_CATCH( std::runtime_error& error ) {
#if TBB_USE_EXCEPTIONS
REPORT("ERROR: %s\n", error.what() );
#endif 
}
}

typedef tbb::blocked_range<int> Range;

static tbb::enumerable_thread_specific<int> local_id, old_id, slot_id(-3);

void ResetTLS() {
local_id.clear();
old_id.clear();
slot_id.clear();
}

class ArenaObserver : public tbb::task_scheduler_observer {
int myId;               
int myMaxConcurrency;   
int myNumReservedSlots; 
void on_scheduler_entry( bool is_worker ) __TBB_override {
int current_index = tbb::this_task_arena::current_thread_index();
REMARK("a %s #%p is entering arena %d from %d on slot %d\n", is_worker?"worker":"master",
&local_id.local(), myId, local_id.local(), current_index );
ASSERT(current_index<(myMaxConcurrency>1?myMaxConcurrency:2), NULL);
if(is_worker) ASSERT(current_index>=myNumReservedSlots, NULL);

ASSERT(!old_id.local(), "double call to on_scheduler_entry");
old_id.local() = local_id.local();
ASSERT(old_id.local() != myId, "double entry to the same arena");
local_id.local() = myId;
slot_id.local() = current_index;
}
void on_scheduler_exit( bool is_worker ) __TBB_override {
REMARK("a %s #%p is leaving arena %d to %d\n", is_worker?"worker":"master",
&local_id.local(), myId, old_id.local());
ASSERT(local_id.local() == myId, "nesting of arenas is broken");
ASSERT(slot_id.local() == tbb::this_task_arena::current_thread_index(), NULL);
ASSERT(slot_id.local() == tbb::task_arena::current_thread_index(), NULL);
slot_id.local() = -2;
local_id.local() = old_id.local();
old_id.local() = 0;
}
public:
ArenaObserver(tbb::task_arena &a, int maxConcurrency, int numReservedSlots, int id)
: tbb::task_scheduler_observer(a)
, myId(id)
, myMaxConcurrency(maxConcurrency)
, myNumReservedSlots(numReservedSlots) {
ASSERT(myId, NULL);
observe(true);
}
~ArenaObserver () {
ASSERT(!old_id.local(), "inconsistent observer state");
}
};

struct IndexTrackingBody { 
void operator() ( const Range& ) const {
ASSERT(slot_id.local() == tbb::this_task_arena::current_thread_index(), NULL);
ASSERT(slot_id.local() == tbb::task_arena::current_thread_index(), NULL);
for ( volatile int i = 0; i < 50000; ++i )
;
}
};

struct AsynchronousWork : NoAssign {
Harness::SpinBarrier &my_barrier;
bool my_is_blocking;
AsynchronousWork(Harness::SpinBarrier &a_barrier, bool blocking = true)
: my_barrier(a_barrier), my_is_blocking(blocking) {}
void operator()() const {
ASSERT(local_id.local() != 0, "not in explicit arena");
tbb::parallel_for(Range(0,500), IndexTrackingBody(), tbb::simple_partitioner(), *tbb::task::self().group());
if(my_is_blocking) my_barrier.timed_wait(10); 
else my_barrier.signal_nowait();
}
};

void TestConcurrentArenasFunc(int idx) {
struct LocalObserver : public tbb::task_scheduler_observer {
LocalObserver() : tbb::task_scheduler_observer(true) { observe(true); }
};
tbb::task_arena a1;
a1.initialize(1,0);
ArenaObserver o1(a1, 1, 0, idx*2+1); 
ASSERT(o1.is_observing(), "Arena observer has not been activated");
LocalObserver lo;
ASSERT(lo.is_observing(), "Local observer has not been activated");
tbb::task_arena a2(2,1);
ArenaObserver o2(a2, 2, 1, idx*2+2);
ASSERT(o2.is_observing(), "Arena observer has not been activated");
Harness::SpinBarrier barrier(2);
AsynchronousWork work(barrier);
a1.enqueue(work); 
barrier.timed_wait(10);
a2.enqueue(work); 
a2.execute(work); 
a1.debug_wait_until_empty();
a2.debug_wait_until_empty();
}

void TestConcurrentArenas(int p) {
ResetTLS();
NativeParallelFor( p, &TestConcurrentArenasFunc );
}

class MultipleMastersPart1 : NoAssign {
tbb::task_arena &my_a;
Harness::SpinBarrier &my_b1, &my_b2;
public:
MultipleMastersPart1( tbb::task_arena &a, Harness::SpinBarrier &b1, Harness::SpinBarrier &b2)
: my_a(a), my_b1(b1), my_b2(b2) {}
void operator()(int) const {
my_a.execute(AsynchronousWork(my_b2, false));
my_b1.timed_wait(10);
my_a.enqueue(AsynchronousWork(my_b2, false));
}
};

class MultipleMastersPart2 : NoAssign {
tbb::task_arena &my_a;
Harness::SpinBarrier &my_b;
public:
MultipleMastersPart2( tbb::task_arena &a, Harness::SpinBarrier &b) : my_a(a), my_b(b) {}
void operator()(int) const {
my_a.execute(AsynchronousWork(my_b, false));
}
};

class MultipleMastersPart3 : NoAssign {
tbb::task_arena &my_a;
Harness::SpinBarrier &my_b;

struct Runner : NoAssign {
tbb::task* const a_task;
Runner(tbb::task* const t) : a_task(t) {}
void operator()() const {
for ( volatile int i = 0; i < 10000; ++i )
;
a_task->decrement_ref_count();
}
};

struct Waiter : NoAssign {
tbb::task* const a_task;
Waiter(tbb::task* const t) : a_task(t) {}
void operator()() const {
a_task->wait_for_all();
}
};

public:
MultipleMastersPart3(tbb::task_arena &a, Harness::SpinBarrier &b)
: my_a(a), my_b(b) {}
void operator()(int idx) const {
tbb::empty_task* root_task = new(tbb::task::allocate_root()) tbb::empty_task;
my_b.timed_wait(10); 
for( int i=0; i<100; ++i) {
root_task->set_ref_count(2);
my_a.enqueue(Runner(root_task));
my_a.execute(Waiter(root_task));
}
tbb::task::destroy(*root_task);
REMARK("Master #%d: job completed, wait for others\n", idx);
my_b.timed_wait(10);
}
};

class MultipleMastersPart4 : NoAssign {
tbb::task_arena &my_a;
Harness::SpinBarrier &my_b;
tbb::task_group_context *my_ag;

struct Getter : NoAssign {
tbb::task_group_context *& my_g;
Getter(tbb::task_group_context *&a_g) : my_g(a_g) {}
void operator()() const {
my_g = tbb::task::self().group();
}
};
struct Checker : NoAssign {
tbb::task_group_context *my_g;
Checker(tbb::task_group_context *a_g) : my_g(a_g) {}
void operator()() const {
ASSERT(my_g == tbb::task::self().group(), NULL);
tbb::task *t = new( tbb::task::allocate_root() ) tbb::empty_task;
ASSERT(my_g == t->group(), NULL);
tbb::task::destroy(*t);
}
};
struct NestedChecker : NoAssign {
const MultipleMastersPart4 &my_body;
NestedChecker(const MultipleMastersPart4 &b) : my_body(b) {}
void operator()() const {
tbb::task_group_context *nested_g = tbb::task::self().group();
ASSERT(my_body.my_ag != nested_g, NULL);
tbb::task *t = new( tbb::task::allocate_root() ) tbb::empty_task;
ASSERT(nested_g == t->group(), NULL);
tbb::task::destroy(*t);
my_body.my_a.enqueue(Checker(my_body.my_ag));
}
};
public:
MultipleMastersPart4( tbb::task_arena &a, Harness::SpinBarrier &b) : my_a(a), my_b(b) {
my_a.execute(Getter(my_ag));
}
void operator()(int) const {
my_a.execute(*this);
}
void operator()() const {
Checker check(my_ag);
check();
tbb::task_arena nested(1,1);
nested.execute(NestedChecker(*this)); 
tbb::parallel_for(Range(0,1),*this); 
my_b.timed_wait(10);
my_a.execute(check);
check();
}
void operator()(const Range &) const {
NestedChecker(*this)();
my_a.execute(Checker(my_ag)); 
}
};

void TestMultipleMasters(int p) {
{
REMARK("multiple masters, part 1\n");
ResetTLS();
tbb::task_arena a(1,0);
a.initialize();
ArenaObserver o(a, 1, 0, 1);
Harness::SpinBarrier barrier1(p), barrier2(2*p+1); 
NativeParallelFor( p, MultipleMastersPart1(a, barrier1, barrier2) );
barrier2.timed_wait(10);
a.debug_wait_until_empty();
} {
REMARK("multiple masters, part 2\n");
ResetTLS();
tbb::task_arena a(2,1);
ArenaObserver o(a, 2, 1, 2);
Harness::SpinBarrier barrier(p+2);
a.enqueue(AsynchronousWork(barrier, true)); 
NativeParallelFor( p, MultipleMastersPart2(a, barrier) );
barrier.timed_wait(10);
a.debug_wait_until_empty();
} {
REMARK("multiple masters, part 3: wait_for_all() in execute()\n");
tbb::task_arena a(p,1);
Harness::SpinBarrier barrier(p+1); 
NativeParallelFor( p+1, MultipleMastersPart3(a, barrier) );
a.debug_wait_until_empty();
} {
int c = p%3? (p%2? p : 2) : 3;
REMARK("multiple masters, part 4: contexts, arena(%d)\n", c);
ResetTLS();
tbb::task_arena a(c, 1);
ArenaObserver o(a, c, 1, c);
Harness::SpinBarrier barrier(c);
MultipleMastersPart4 test(a, barrier);
NativeParallelFor(p, test);
a.debug_wait_until_empty();
}
}

#include <sstream>
#if TBB_USE_EXCEPTIONS
#include <stdexcept>
#include "tbb/tbb_exception.h"
#endif

struct TestArenaEntryBody : FPModeContext {
tbb::atomic<int> &my_stage; 
std::stringstream my_id;
bool is_caught, is_expected;
enum { arenaFPMode = 1 };

TestArenaEntryBody(tbb::atomic<int> &s, int idx, int i)  
:   FPModeContext(idx+i)
,   my_stage(s)
,   is_caught(false)
,   is_expected( (idx&(1<<i)) != 0 && (TBB_USE_EXCEPTIONS) != 0 )
{
my_id << idx << ':' << i << '@';
}
void operator()() { 
int stage = my_stage++;
int slot = tbb::this_task_arena::current_thread_index();
ASSERT(slot >= 0 && slot <= 1, "master or the only worker");
while(my_stage < 2+slot) __TBB_Yield();
my_id << (stage < 3 ? (tbb::this_task_arena::current_thread_index()?
"delegated_to_worker" : stage < 2? "direct" : "delegated_to_master")
: stage == 3? "nested_same_ctx" : "nested_alien_ctx");
REMARK("running %s\n", my_id.str().c_str());
AssertFPMode(arenaFPMode);
if(is_expected)
__TBB_THROW(std::logic_error(my_id.str()));
}
void on_exception(const char *e) { 
is_caught = true;
REMARK("caught %s\n", e);
ASSERT(my_id.str() == e, NULL);
assertFPMode();
}
void after_execute() { 
REMARK("completing %s\n", my_id.str().c_str() );
ASSERT(is_caught == is_expected, NULL);
assertFPMode();
}
};

class ForEachArenaEntryBody : NoAssign {
tbb::task_arena &my_a; 
tbb::atomic<int> &my_stage; 
int my_idx;

public:
ForEachArenaEntryBody(tbb::task_arena &a, tbb::atomic<int> &c)
: my_a(a), my_stage(c), my_idx(0) {}

void test(int idx) {
my_idx = idx;
my_stage = 0;
NativeParallelFor(3, *this); 
ASSERT(my_stage == 3, NULL);
my_a.execute(*this); 
ASSERT(my_stage == 5, NULL);
}

void operator()() const {
test_arena_entry(3); 
tbb::parallel_for(4, 5, *this); 
}

void operator()(int i) const {
test_arena_entry(i);
}

private:
void test_arena_entry(int i) const {
TestArenaEntryBody scoped_functor(my_stage, my_idx, i);
__TBB_TRY {
my_a.execute(scoped_functor);
}
#if TBB_USE_EXCEPTIONS
catch(tbb::captured_exception &e) {
scoped_functor.on_exception(e.what());
ASSERT_WARNING(TBB_USE_CAPTURED_EXCEPTION, "Caught captured_exception while expecting exact one");
} catch(std::logic_error &e) {
scoped_functor.on_exception(e.what());
ASSERT(!TBB_USE_CAPTURED_EXCEPTION, "Caught exception of wrong type");
} catch(...) { ASSERT(false, "Unexpected exception type"); }
#endif 
scoped_functor.after_execute();
}
};

void TestArenaEntryConsistency() {
REMARK("test arena entry consistency\n");

tbb::task_arena a(2, 1);
tbb::atomic<int> c;
ForEachArenaEntryBody body(a, c);

FPModeContext fp_scope(TestArenaEntryBody::arenaFPMode);
a.initialize(); 
fp_scope.setNextFPMode();

for (int i = 0; i < 100; i++) 
body.test(i);
}

class TestArenaConcurrencyBody : NoAssign {
tbb::task_arena &my_a;
int my_max_concurrency;
int my_reserved_slots;
Harness::SpinBarrier *my_barrier;
Harness::SpinBarrier *my_worker_barrier;
public:
TestArenaConcurrencyBody( tbb::task_arena &a, int max_concurrency, int reserved_slots, Harness::SpinBarrier *b = NULL, Harness::SpinBarrier *wb = NULL )
: my_a(a), my_max_concurrency(max_concurrency), my_reserved_slots(reserved_slots), my_barrier(b), my_worker_barrier(wb) {}
void operator()( int ) const {
ASSERT( local_id.local() == 0, "TLS was not cleaned?" );
local_id.local() = 1;
my_a.execute( *this );
}
void operator()() const {
ASSERT( tbb::task_arena::current_thread_index() == tbb::this_task_arena::current_thread_index(), NULL );
int idx = tbb::this_task_arena::current_thread_index();
ASSERT( idx < (my_max_concurrency > 1 ? my_max_concurrency : 2), NULL );
ASSERT( my_a.max_concurrency() == tbb::this_task_arena::max_concurrency(), NULL );
int max_arena_concurrency = tbb::this_task_arena::max_concurrency();
ASSERT( max_arena_concurrency == my_max_concurrency, NULL );
if ( my_worker_barrier ) {
if ( local_id.local() == 1 ) {
ASSERT( idx < my_reserved_slots, "Masters are supposed to use only reserved slots in this test" );
} else {
ASSERT( idx >= my_reserved_slots, NULL );
my_worker_barrier->timed_wait( 10 );
}
} else if ( my_barrier )
ASSERT( local_id.local() == 1, "Workers are not supposed to enter the arena in this test" );
if ( my_barrier ) my_barrier->timed_wait( 10 );
else Harness::Sleep( 10 );
}
};

void TestArenaConcurrency( int p, int reserved = 0, int step = 1) {
for (; reserved <= p; reserved += step) {
REMARK("TestArenaConcurrency: %d slots, %d reserved\n", p, reserved);
tbb::task_arena a( p, reserved );
{ 
ResetTLS();
Harness::SpinBarrier b( p );
Harness::SpinBarrier wb( p-reserved );
TestArenaConcurrencyBody test( a, p, reserved, &b, &wb );
for ( int i = reserved; i < p; ++i )
a.enqueue( test );
if ( reserved==1 )
test( 0 ); 
else
NativeParallelFor( reserved, test );
a.debug_wait_until_empty();
} { 
ResetTLS();
Harness::SpinBarrier b( p );
NativeParallelFor( p, TestArenaConcurrencyBody( a, p, reserved, &b ) );
a.debug_wait_until_empty();
} { 
ResetTLS();
NativeParallelFor( 2*p, TestArenaConcurrencyBody( a, p, reserved ) );
a.debug_wait_until_empty();
}
}
}


typedef tbb::interface7::internal::task_arena_base task_arena_internals;

struct TaskArenaValidator : public task_arena_internals {
int my_slot_at_construction;
TaskArenaValidator( const task_arena_internals& other )
: task_arena_internals(other)  {
my_slot_at_construction = tbb::this_task_arena::current_thread_index();
}
int concurrency() { return my_max_concurrency; }
int reserved_for_masters() { return (int)my_master_slots; }

void operator()() {
ASSERT( tbb::this_task_arena::current_thread_index()==my_slot_at_construction,
"Current thread index has changed since the validator construction" );
ASSERT( tbb::task_arena::current_thread_index()==my_slot_at_construction,
"Current thread index has changed since the validator construction" );
}
};

void ValidateAttachedArena( tbb::task_arena& arena, bool expect_activated,
int expect_concurrency, int expect_masters ) {
ASSERT( arena.is_active()==expect_activated, "Unexpected activation state" );
if( arena.is_active() ) {
TaskArenaValidator validator( arena );
ASSERT( validator.concurrency()==expect_concurrency, "Unexpected arena size" );
ASSERT( validator.reserved_for_masters()==expect_masters, "Unexpected # of reserved slots" );
if ( tbb::this_task_arena::current_thread_index() != tbb::task_arena::not_initialized ) {
ASSERT( tbb::task_arena::current_thread_index() >= 0 &&
tbb::this_task_arena::current_thread_index() >= 0, NULL);
arena.execute( validator );
} else { 
ASSERT( tbb::task_arena::current_thread_index()==-1, NULL);
}

}
}

struct TestAttachBody : NoAssign {
mutable int my_idx; 
const int maxthread;
TestAttachBody( int max_thr ) : maxthread(max_thr) {}

void operator()( int idx ) const {
my_idx = idx;
int default_threads = tbb::task_scheduler_init::default_num_threads();

tbb::task_arena arena = tbb::task_arena( tbb::task_arena::attach() );
ValidateAttachedArena( arena, false, -1, -1 ); 

{ 
tbb::task_scheduler_init init( idx+1 );

tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
ValidateAttachedArena( arena2, true, idx+1, 1 );

arena.initialize( tbb::task_arena::attach() );
}
ValidateAttachedArena( arena, true, idx+1, 1 );

arena.terminate();
ValidateAttachedArena( arena, false, -1, -1 );

switch (idx%2) {
case 0:
{ 
tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
ValidateAttachedArena( arena2, false, -1, -1 );
arena2.initialize(); 
ValidateAttachedArena( arena2, true, default_threads, 1 );
}
break;
case 1:
{ 
tbb::task_arena arena2;
ValidateAttachedArena( arena2, false, -1, -1 );
arena2.initialize( tbb::task_arena::attach() ); 
ValidateAttachedArena( arena2, true, default_threads, 1 );
}
break;
} 

tbb::empty_task& tsk = *new (tbb::task::allocate_root()) tbb::empty_task;
tbb::task::spawn_root_and_wait(tsk);
tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
ValidateAttachedArena( arena2, true, default_threads, 1 );

arena.initialize( maxthread, min(maxthread,idx) );
arena.execute( *this );
}

void operator()() const {
tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
ValidateAttachedArena( arena2, true, maxthread, min(maxthread,my_idx) );
}

void operator()( const Range& r ) const {
for( int i = r.begin(); i<r.end(); ++i ) {
tbb::task_arena arena2 = tbb::task_arena( tbb::task_arena::attach() );
ValidateAttachedArena( arena2, true, maxthread+1, 1 ); 
}
}
};

void TestAttach( int maxthread ) {
REMARK( "Testing attached task_arenas\n" );
NativeParallelFor( max(maxthread,4), TestAttachBody( maxthread ) );
tbb::parallel_for( Range(0,10000*maxthread), TestAttachBody( maxthread ) );
}

struct TestFunctor {
void operator()() { ASSERT( false, "Non-const operator called" ); }
void operator()() const {  }
};

void TestConstantFunctorRequirement() {
tbb::task_arena a;
TestFunctor tf;
a.enqueue( tf );
#if __TBB_TASK_PRIORITY
a.enqueue( tf, tbb::priority_normal );
#endif
}
#if __TBB_TASK_ISOLATION
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_invoke.h"
namespace TestIsolatedExecuteNS {
template <typename NestedPartitioner>
class NestedParFor : NoAssign {
public:
NestedParFor() {}
void operator()() const {
NestedPartitioner p;
tbb::parallel_for( 0, 10, Harness::DummyBody( 10 ), p );
}
};

template <typename NestedPartitioner>
class ParForBody : NoAssign {
bool myOuterIsolation;
tbb::enumerable_thread_specific<int> &myEts;
tbb::atomic<bool> &myIsStolen;
public:
ParForBody( bool outer_isolation, tbb::enumerable_thread_specific<int> &ets, tbb::atomic<bool> &is_stolen )
: myOuterIsolation( outer_isolation ), myEts( ets ), myIsStolen( is_stolen ) {}
void operator()( int ) const {
int &e = myEts.local();
if ( e++ > 0 ) myIsStolen = true;
if ( myOuterIsolation )
NestedParFor<NestedPartitioner>()();
else
tbb::this_task_arena::isolate( NestedParFor<NestedPartitioner>() );
--e;
}
};

template <typename OuterPartitioner, typename NestedPartitioner>
class OuterParFor : NoAssign {
bool myOuterIsolation;
tbb::atomic<bool> &myIsStolen;
public:
OuterParFor( bool outer_isolation, tbb::atomic<bool> &is_stolen ) : myOuterIsolation( outer_isolation ), myIsStolen( is_stolen ) {}
void operator()() const {
tbb::enumerable_thread_specific<int> ets( 0 );
OuterPartitioner p;
tbb::parallel_for( 0, 1000, ParForBody<NestedPartitioner>( myOuterIsolation, ets, myIsStolen ), p );
}
};

template <typename OuterPartitioner, typename NestedPartitioner>
void TwoLoopsTest( bool outer_isolation ) {
tbb::atomic<bool> is_stolen;
is_stolen = false;
const int max_repeats = 100;
if ( outer_isolation ) {
for ( int i = 0; i <= max_repeats; ++i ) {
tbb::this_task_arena::isolate( OuterParFor<OuterPartitioner, NestedPartitioner>( outer_isolation, is_stolen ) );
if ( is_stolen ) break;
}
ASSERT_WARNING( is_stolen, "isolate() should not block stealing on nested levels without isolation" );
} else {
for ( int i = 0; i <= max_repeats; ++i ) {
OuterParFor<OuterPartitioner, NestedPartitioner>( outer_isolation, is_stolen )();
}
ASSERT( !is_stolen, "isolate() on nested levels should prevent stealing from outer leves" );
}
}

void TwoLoopsTest( bool outer_isolation ) {
TwoLoopsTest<tbb::simple_partitioner, tbb::simple_partitioner>( outer_isolation );
TwoLoopsTest<tbb::simple_partitioner, tbb::affinity_partitioner>( outer_isolation );
TwoLoopsTest<tbb::affinity_partitioner, tbb::simple_partitioner>( outer_isolation );
TwoLoopsTest<tbb::affinity_partitioner, tbb::affinity_partitioner>( outer_isolation );
}

void TwoLoopsTest() {
TwoLoopsTest( true );
TwoLoopsTest( false );
}
class HeavyMixTestBody : NoAssign {
tbb::enumerable_thread_specific<Harness::FastRandom>& myRandom;
tbb::enumerable_thread_specific<int>& myIsolatedLevel;
int myNestedLevel;

template <typename Partitioner, typename Body>
static void RunTwoBodies( Harness::FastRandom& rnd, const Body &body, Partitioner& p, tbb::task_group_context* ctx = NULL ) {
if ( rnd.get() % 2 )
if  (ctx )
tbb::parallel_for( 0, 2, body, p, *ctx );
else
tbb::parallel_for( 0, 2, body, p );
else
tbb::parallel_invoke( body, body );
}

template <typename Partitioner>
class IsolatedBody : NoAssign {
const HeavyMixTestBody &myHeavyMixTestBody;
Partitioner &myPartitioner;
public:
IsolatedBody( const HeavyMixTestBody &body, Partitioner &partitioner )
: myHeavyMixTestBody( body ), myPartitioner( partitioner ) {}
void operator()() const {
RunTwoBodies( myHeavyMixTestBody.myRandom.local(),
HeavyMixTestBody( myHeavyMixTestBody.myRandom, myHeavyMixTestBody.myIsolatedLevel,
myHeavyMixTestBody.myNestedLevel + 1 ),
myPartitioner );
}
};

template <typename Partitioner>
void RunNextLevel( Harness::FastRandom& rnd, int &isolated_level ) const {
Partitioner p;
switch ( rnd.get() % 3 ) {
case 0: {
tbb::task_group_context ctx;
RunTwoBodies( rnd, HeavyMixTestBody(myRandom, myIsolatedLevel, myNestedLevel + 1), p, &ctx );
break;
}
case 1: {
tbb::task_group_context ctx;
#if __TBB_TASK_PRIORITY
ctx.set_priority( tbb::priority_high );
#endif
RunTwoBodies( rnd, HeavyMixTestBody(myRandom, myIsolatedLevel, myNestedLevel + 1), p, &ctx );
break;
}
case 2: {
int previous_isolation = isolated_level;
isolated_level = myNestedLevel;
tbb::this_task_arena::isolate( IsolatedBody<Partitioner>( *this, p ) );
isolated_level = previous_isolation;
break;
}
}
}
public:
HeavyMixTestBody( tbb::enumerable_thread_specific<Harness::FastRandom>& random,
tbb::enumerable_thread_specific<int>& isolated_level, int nested_level )
: myRandom( random ), myIsolatedLevel( isolated_level )
, myNestedLevel( nested_level ) {}
void operator()() const {
int &isolated_level = myIsolatedLevel.local();
ASSERT( myNestedLevel > isolated_level, "The outer-level task should not be stolen on isolated level" );
if ( myNestedLevel == 20 )
return;
Harness::FastRandom &rnd = myRandom.local();
if ( rnd.get() % 2 == 1 ) {
RunNextLevel<tbb::auto_partitioner>( rnd, isolated_level );
} else {
RunNextLevel<tbb::affinity_partitioner>( rnd, isolated_level );
}
}
void operator()(int) const {
this->operator()();
}
};

struct RandomInitializer {
Harness::FastRandom operator()() {
return Harness::FastRandom( tbb::this_task_arena::current_thread_index() );
}
};

void HeavyMixTest() {
tbb::task_scheduler_init init( tbb::task_scheduler_init::default_num_threads() < 3 ? 3 : tbb::task_scheduler_init::automatic );
RandomInitializer init_random;
tbb::enumerable_thread_specific<Harness::FastRandom> random( init_random );
tbb::enumerable_thread_specific<int> isolated_level( 0 );
for ( int i = 0; i < 5; ++i ) {
HeavyMixTestBody b( random, isolated_level, 1 );
b( 0 );
REMARK( "." );
}
}
struct ContinuationTestReduceBody : NoAssign {
tbb::internal::isolation_tag myIsolation;
ContinuationTestReduceBody( tbb::internal::isolation_tag isolation ) : myIsolation( isolation ) {}
ContinuationTestReduceBody( ContinuationTestReduceBody& b, tbb::split ) : myIsolation( b.myIsolation ) {}
void operator()( tbb::blocked_range<int> ) {}
void join( ContinuationTestReduceBody& ) {
tbb::internal::isolation_tag isolation = tbb::task::self().prefix().isolation;
ASSERT( isolation == myIsolation, "The continuations should preserve children's isolation" );
}
};
struct ContinuationTestIsolated {
void operator()() const {
ContinuationTestReduceBody b( tbb::task::self().prefix().isolation );
tbb::parallel_deterministic_reduce( tbb::blocked_range<int>( 0, 100 ), b );
}
};
struct ContinuationTestParForBody : NoAssign {
tbb::enumerable_thread_specific<int> &myEts;
public:
ContinuationTestParForBody( tbb::enumerable_thread_specific<int> &ets ) : myEts( ets ){}
void operator()( int ) const {
int &e = myEts.local();
++e;
ASSERT( e==1, "The task is stolen on isolated level" );
tbb::this_task_arena::isolate( ContinuationTestIsolated() );
--e;
}
};
void ContinuationTest() {
for ( int i = 0; i < 5; ++i ) {
tbb::enumerable_thread_specific<int> myEts;
tbb::parallel_for( 0, 100, ContinuationTestParForBody( myEts ), tbb::simple_partitioner() );
}
}
#if TBB_USE_EXCEPTIONS
struct MyException {};
struct IsolatedBodyThrowsException {
void operator()() const {
__TBB_THROW( MyException() );
}
};
struct ExceptionTestBody : NoAssign {
tbb::enumerable_thread_specific<int>& myEts;
tbb::atomic<bool>& myIsStolen;
ExceptionTestBody( tbb::enumerable_thread_specific<int>& ets, tbb::atomic<bool>& is_stolen )
: myEts( ets ), myIsStolen( is_stolen ) {}
void operator()( int i ) const {
try {
tbb::this_task_arena::isolate( IsolatedBodyThrowsException() );
ASSERT( false, "The exception has been lost" );
}
catch ( MyException ) {}
catch ( ... ) {
ASSERT( false, "Unexpected exception" );
}
int &e = myEts.local();
if ( e++ > 0 ) myIsStolen = true;
tbb::parallel_for( 0, 10+i, Harness::DummyBody( 100 ) );
--e;
}
};

#endif 
void ExceptionTest() {
#if TBB_USE_EXCEPTIONS
tbb::enumerable_thread_specific<int> ets;
tbb::atomic<bool> is_stolen;
is_stolen = false;
for ( int i = 0; i<10; ++i ) {
tbb::parallel_for( 0, 1000, ExceptionTestBody( ets, is_stolen ) );
if ( is_stolen ) break;
}
ASSERT( is_stolen, "isolate should not affect non-isolated work" );
#endif 
}

struct NonConstBody {
unsigned int state;
void operator()() {
state ^= ~0u;
}
};

void TestNonConstBody() {
NonConstBody body;
body.state = 0x6c97d5ed;
tbb::this_task_arena::isolate(body);
ASSERT(body.state == 0x93682a12, "The wrong state");
}

class TestEnqueueTask : public tbb::task {
bool enqueued;
tbb::enumerable_thread_specific<bool>& executed;
tbb::atomic<int>& completed;
public:
static const int N = 100;

TestEnqueueTask(bool enq, tbb::enumerable_thread_specific<bool>& exe, tbb::atomic<int>& c)
: enqueued(enq), executed(exe), completed(c) {}
tbb::task* execute() __TBB_override {
if (enqueued) {
executed.local() = true;
++completed;
__TBB_Yield();
} else {
parent()->add_ref_count(N);
for (int i = 0; i < N; ++i)
tbb::task::enqueue(*new (parent()->allocate_child()) TestEnqueueTask(true, executed, completed));
}
return NULL;
}
};

class TestEnqueueIsolateBody : NoCopy {
tbb::enumerable_thread_specific<bool>& executed;
tbb::atomic<int>& completed;
public:
TestEnqueueIsolateBody(tbb::enumerable_thread_specific<bool>& exe, tbb::atomic<int>& c)
: executed(exe), completed(c) {}
void operator()() {
tbb::task::spawn_root_and_wait(*new (tbb::task::allocate_root()) TestEnqueueTask(false, executed, completed));
}
};

void TestEnqueue() {
tbb::enumerable_thread_specific<bool> executed(false);
tbb::atomic<int> completed;

completed = 0;
TestEnqueueIsolateBody b1(executed, completed);
b1();
if (!executed.local())
REPORT("Warning: No one enqueued task has executed by the main thread.\n");

executed.local() = false;
completed = 0;
const int N = 100;
for (int i = 0; i < N; ++i)
tbb::task::enqueue(*new (tbb::task::allocate_root()) TestEnqueueTask(true, executed, completed));
TestEnqueueIsolateBody b2(executed, completed);
tbb::this_task_arena::isolate(b2);
ASSERT(executed.local() == false, "An enqueued task was executed within isolate.");

while (completed < TestEnqueueTask::N + N) __TBB_Yield();
}
}

void TestIsolatedExecute() {
REMARK("TestIsolatedExecute");
int num_threads = min( tbb::task_scheduler_init::default_num_threads(), 3 );
{
tbb::task_scheduler_init init( max(num_threads, 7) );
REMARK("."); TestIsolatedExecuteNS::TwoLoopsTest();
REMARK("."); TestIsolatedExecuteNS::HeavyMixTest();
REMARK("."); TestIsolatedExecuteNS::ContinuationTest();
REMARK("."); TestIsolatedExecuteNS::ExceptionTest();
}
tbb::task_scheduler_init init(num_threads);
REMARK("."); TestIsolatedExecuteNS::HeavyMixTest();
REMARK("."); TestIsolatedExecuteNS::ContinuationTest();
REMARK("."); TestIsolatedExecuteNS::TestNonConstBody();
REMARK("."); TestIsolatedExecuteNS::TestEnqueue();
REMARK("\rTestIsolatedExecute: done                                                  \n");
}
#endif 

class TestDelegatedSpawnWaitBody : NoAssign {
tbb::task_arena &my_a;
Harness::SpinBarrier &my_b1, &my_b2;

struct Spawner : NoAssign {
tbb::task* const a_task;
Spawner(tbb::task* const t) : a_task(t) {}
void operator()() const {
tbb::task::spawn( *new(a_task->allocate_child()) tbb::empty_task );
}
};

struct Waiter : NoAssign {
tbb::task* const a_task;
Waiter(tbb::task* const t) : a_task(t) {}
void operator()() const {
a_task->wait_for_all();
}
};

public:
TestDelegatedSpawnWaitBody( tbb::task_arena &a, Harness::SpinBarrier &b1, Harness::SpinBarrier &b2)
: my_a(a), my_b1(b1), my_b2(b2) {}
void operator()(int idx) const {
if ( idx==0 ) { 
for( int i=0; i<2; ++i ) my_a.enqueue(*this); 
tbb::empty_task* root_task = new(tbb::task::allocate_root()) tbb::empty_task;
root_task->set_ref_count(100001);
my_b1.timed_wait(10); 
for( int i=0; i<100000; ++i) {
my_a.execute(Spawner(root_task));
}
my_a.execute(Waiter(root_task));
tbb::task::destroy(*root_task);
}
my_b2.timed_wait(10); 
}
void operator()() const {
my_b1.timed_wait(10); 
}
};

void TestDelegatedSpawnWait() {
REMARK( "Testing delegated spawn & wait\n" );
tbb::task_arena a(2,0);
a.initialize();
Harness::SpinBarrier barrier1(3), barrier2(2);
NativeParallelFor( 2, TestDelegatedSpawnWaitBody(a, barrier1, barrier2) );
a.debug_wait_until_empty();
}

class TestMultipleWaitsArenaWait : NoAssign {
public:
TestMultipleWaitsArenaWait( int idx, int bunch_size, int num_tasks, tbb::task** waiters, tbb::atomic<int>& processed )
: my_idx( idx ), my_bunch_size( bunch_size ), my_num_tasks(num_tasks), my_waiters( waiters ), my_processed( processed ) {}
void operator()() const {
++my_processed;
if ( my_idx < my_num_tasks )
my_waiters[my_idx]->wait_for_all();
if ( my_idx >= my_bunch_size )
my_waiters[my_idx-my_bunch_size]->decrement_ref_count();
}
private:
int my_idx;
int my_bunch_size;
int my_num_tasks;
tbb::task** my_waiters;
tbb::atomic<int>& my_processed;
};

class TestMultipleWaitsThreadBody : NoAssign {
public:
TestMultipleWaitsThreadBody( int bunch_size, int num_tasks, tbb::task_arena& a, tbb::task** waiters, tbb::atomic<int>& processed )
: my_bunch_size( bunch_size ), my_num_tasks( num_tasks ), my_arena( a ), my_waiters( waiters ), my_processed( processed ) {}
void operator()( int idx ) const {
my_arena.execute( TestMultipleWaitsArenaWait( idx, my_bunch_size, my_num_tasks, my_waiters, my_processed ) );
--my_processed;
}
private:
int my_bunch_size;
int my_num_tasks;
tbb::task_arena& my_arena;
tbb::task** my_waiters;
tbb::atomic<int>& my_processed;
};

#include "tbb/tbb_thread.h"

void TestMultipleWaits( int num_threads, int num_bunches, int bunch_size ) {
tbb::task_arena a( num_threads );
const int num_tasks = (num_bunches-1)*bunch_size;
tbb::task** tasks = new tbb::task*[num_tasks];
for ( int i = 0; i<num_tasks; ++i )
tasks[i] = new (tbb::task::allocate_root()) tbb::empty_task();
tbb::atomic<int> processed;
processed = 0;
for ( int repeats = 0; repeats<10; ++repeats ) {
int idx = 0;
for ( int bunch = 0; bunch < num_bunches-1; ++bunch ) {
while ( processed < bunch*bunch_size ) __TBB_Yield();
for ( int i = 0; i<bunch_size; ++i ) {
tasks[idx]->set_ref_count( 2 );
tbb::tbb_thread( TestMultipleWaitsThreadBody( bunch_size, num_tasks, a, tasks, processed ), idx++ ).detach();
}
}
for ( int i = 0; i<bunch_size; ++i )
tbb::tbb_thread( TestMultipleWaitsThreadBody( bunch_size, num_tasks, a, tasks, processed ), idx++ ).detach();
while ( processed ) __TBB_Yield();
}
for ( int i = 0; i<num_tasks; ++i )
tbb::task::destroy( *tasks[i] );
delete[] tasks;
}

void TestMultipleWaits() {
REMARK( "Testing multiple waits\n" );
const int max_threads = min( 16, tbb::task_scheduler_init::default_num_threads() );

Harness::FastRandom rnd(1234);
for ( int threads = 1; threads <= max_threads; threads += max( threads/2, 1 ) ) {
for ( int i = 0; i<3; ++i ) {
const int num_bunches = 3 + rnd.get()%3;
const int bunch_size = max_threads + rnd.get()%max_threads;
TestMultipleWaits( threads, num_bunches, bunch_size );
}
}
}
#include "tbb/global_control.h"

void TestSmallStackSize() {
tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic,
tbb::global_control::active_value(tbb::global_control::thread_stack_size) / 2 );
for (int i = 0; i < 100; ++i) {
tbb::task_arena a;
a.initialize();
}
}
#if __TBB_CPP11_RVALUE_REF_PRESENT
namespace TestMoveSemanticsNS {
struct TestFunctor {
void operator()() const {};
};

struct MoveOnlyFunctor : MoveOnly, TestFunctor {
MoveOnlyFunctor() : MoveOnly() {};
MoveOnlyFunctor(MoveOnlyFunctor&& other) : MoveOnly(std::move(other)) {};
};

struct MovePreferableFunctor : Movable, TestFunctor {
MovePreferableFunctor() : Movable() {};
MovePreferableFunctor(MovePreferableFunctor&& other) : Movable( std::move(other) ) {};
MovePreferableFunctor(const MovePreferableFunctor& other) : Movable(other) {};
};

struct NoMoveNoCopyFunctor : NoCopy, TestFunctor {
NoMoveNoCopyFunctor() : NoCopy() {};
private:
NoMoveNoCopyFunctor(NoMoveNoCopyFunctor&&);
};


void TestFunctors() {
tbb::task_arena ta;
MovePreferableFunctor mpf;
ta.execute( NoMoveNoCopyFunctor() );

ta.enqueue( MoveOnlyFunctor() );
ta.enqueue( mpf );
ASSERT(mpf.alive, "object was moved when was passed by lval");
mpf.Reset();
ta.enqueue( std::move(mpf) );
ASSERT(!mpf.alive, "object was copied when was passed by rval");
mpf.Reset();
#if __TBB_TASK_PRIORITY
ta.enqueue( MoveOnlyFunctor(), tbb::priority_normal );
ta.enqueue( mpf, tbb::priority_normal );
ASSERT(mpf.alive, "object was moved when was passed by lval");
mpf.Reset();
ta.enqueue( std::move(mpf), tbb::priority_normal );
ASSERT(!mpf.alive, "object was copied when was passed by rval");
mpf.Reset();
#endif
}
}
#endif 

void TestMoveSemantics() {
#if __TBB_CPP11_RVALUE_REF_PRESENT
TestMoveSemanticsNS::TestFunctors();
#else
REPORT("Known issue: move support tests are skipped.\n");
#endif
}
#if __TBB_CPP11_DECLTYPE_PRESENT && !__TBB_CPP11_DECLTYPE_OF_FUNCTION_RETURN_TYPE_BROKEN
#include <vector>
#include "harness_state_trackable.h"

namespace TestReturnValueNS {
struct noDefaultTag {};
class ReturnType : public Harness::StateTrackable<> {
static const int SIZE = 42;
std::vector<int> data;
public:
ReturnType(noDefaultTag) : Harness::StateTrackable<>(0) {}
#if !__TBB_IMPLICIT_MOVE_PRESENT
ReturnType(const ReturnType& r) : Harness::StateTrackable<>(r), data(r.data) {}
ReturnType(ReturnType&& r) : Harness::StateTrackable<>(std::move(r)), data(std::move(r.data)) {}
#endif
void fill() {
for (int i = 0; i < SIZE; ++i)
data.push_back(i);
}
void check() {
ASSERT(data.size() == unsigned(SIZE), NULL);
for (int i = 0; i < SIZE; ++i)
ASSERT(data[i] == i, NULL);
Harness::StateTrackableCounters::counters_t& cnts = Harness::StateTrackableCounters::counters;
ASSERT((cnts[Harness::StateTrackableBase::DefaultInitialized] == 0), NULL);
ASSERT(cnts[Harness::StateTrackableBase::DirectInitialized] == 1, NULL);
std::size_t copied = cnts[Harness::StateTrackableBase::CopyInitialized];
std::size_t moved = cnts[Harness::StateTrackableBase::MoveInitialized];
ASSERT(cnts[Harness::StateTrackableBase::Destroyed] == copied + moved, NULL);
#if __TBB_CPP11_RVALUE_REF_PRESENT
ASSERT(copied == 0 && moved <=3, NULL);
#else
ASSERT(copied <= 3 && moved == 0, NULL);
#endif
}
};

template <typename R>
R function() {
noDefaultTag tag;
R r(tag);
r.fill();
return r;
}

template <>
void function<void>() {}

template <typename R>
struct Functor {
R operator()() const {
return function<R>();
}
};

tbb::task_arena& arena() {
static tbb::task_arena a;
return a;
}

template <typename F>
void TestExecute(F &f) {
Harness::StateTrackableCounters::reset();
ReturnType r = arena().execute(f);
r.check();
}

template <typename F>
void TestExecute(const F &f) {
Harness::StateTrackableCounters::reset();
ReturnType r = arena().execute(f);
r.check();
}
#if TBB_PREVIEW_TASK_ISOLATION
template <typename F>
void TestIsolate(F &f) {
Harness::StateTrackableCounters::reset();
ReturnType r = tbb::this_task_arena::isolate(f);
r.check();
}

template <typename F>
void TestIsolate(const F &f) {
Harness::StateTrackableCounters::reset();
ReturnType r = tbb::this_task_arena::isolate(f);
r.check();
}
#endif

void Test() {
TestExecute(Functor<ReturnType>());
Functor<ReturnType> f1;
TestExecute(f1);
TestExecute(function<ReturnType>);

arena().execute(Functor<void>());
Functor<void> f2;
arena().execute(f2);
arena().execute(function<void>);
#if TBB_PREVIEW_TASK_ISOLATION
TestIsolate(Functor<ReturnType>());
TestIsolate(f1);
TestIsolate(function<ReturnType>);
tbb::this_task_arena::isolate(Functor<void>());
tbb::this_task_arena::isolate(f2);
tbb::this_task_arena::isolate(function<void>);
#endif
}
}
#endif 

void TestReturnValue() {
#if __TBB_CPP11_DECLTYPE_PRESENT && !__TBB_CPP11_DECLTYPE_OF_FUNCTION_RETURN_TYPE_BROKEN
TestReturnValueNS::Test();
#endif
}
void TestConcurrentFunctionality(int min_thread_num = MinThread, int max_thread_num = MaxThread) {
InitializeAndTerminate(max_thread_num);
for (int p = min_thread_num; p <= max_thread_num; ++p) {
REMARK("testing with %d threads\n", p);
TestConcurrentArenas(p);
TestMultipleMasters(p);
TestArenaConcurrency(p);
}
}
struct DefaultCreatedWorkersAmountBody {
int my_threadnum;
DefaultCreatedWorkersAmountBody(int threadnum) : my_threadnum(threadnum) {}
void operator()(int) const {
ASSERT(my_threadnum == tbb::this_task_arena::max_concurrency(), "concurrency level is not equal specified threadnum");
ASSERT(tbb::this_task_arena::current_thread_index() < tbb::this_task_arena::max_concurrency(), "amount of created threads is more than specified by default");
local_id.local() = 1;
Harness::Sleep(1);
}
};

struct NativeParallelForBody {
int my_thread_num;
int iterations;
NativeParallelForBody(int thread_num, int multiplier = 100) : my_thread_num(thread_num), iterations(multiplier * thread_num) {}
void operator()(int idx) const {
ASSERT(idx == 0, "more than 1 thread is going to reset TLS");
ResetTLS();
tbb::parallel_for(0, iterations, DefaultCreatedWorkersAmountBody(my_thread_num), tbb::simple_partitioner());
ASSERT(local_id.size() == size_t(my_thread_num), "amount of created threads is not equal to default num");
}
};

void TestDefaultCreatedWorkersAmount() {
NativeParallelFor(1, NativeParallelForBody(tbb::task_scheduler_init::default_num_threads()));
}

void TestAbilityToCreateWorkers(int thread_num) {
tbb::task_scheduler_init init_market_with_necessary_amount_plus_one(thread_num);
TestArenaConcurrency(thread_num - 1, 0, int(thread_num / 2.72));
TestArenaConcurrency(thread_num, 1, int(thread_num / 3.14));
}

void TestDefaultWorkersLimit() {
TestDefaultCreatedWorkersAmount();
#ifndef RML_USE_WCRM
TestAbilityToCreateWorkers(256);
#endif
}

struct MyObserver: public tbb::task_scheduler_observer {
tbb::enumerable_thread_specific<tbb::task_arena*>& my_tls;
tbb::task_arena& my_arena;
tbb::atomic<int>& my_failure_counter;
tbb::atomic<int>& my_counter;

MyObserver(tbb::task_arena& a,
tbb::enumerable_thread_specific<tbb::task_arena*>& tls,
tbb::atomic<int>& failure_counter,
tbb::atomic<int>& counter)
: tbb::task_scheduler_observer(a), my_tls(tls), my_arena(a),
my_failure_counter(failure_counter), my_counter(counter) {
observe(true);
}
void on_scheduler_entry(bool worker) __TBB_override {
if (worker) {
++my_counter;
tbb::task_arena*& cur_arena = my_tls.local();
if (cur_arena != 0 && cur_arena != &my_arena) {
++my_failure_counter;
}
cur_arena = &my_arena;
}
}
};

struct MyLoopBody {
Harness::SpinBarrier& m_barrier;
MyLoopBody(Harness::SpinBarrier& b):m_barrier(b) { }
void operator()(int) const {
m_barrier.wait();
}
};

struct TaskForArenaExecute {
Harness::SpinBarrier& m_barrier;
TaskForArenaExecute(Harness::SpinBarrier& b):m_barrier(b) { }
void operator()() const {
tbb::parallel_for(0, tbb::this_task_arena::max_concurrency(),
MyLoopBody(m_barrier), tbb::simple_partitioner()
);
}
};

struct ExecuteParallelFor {
int n_per_thread;
int n_repetitions;
std::vector<tbb::task_arena>& arenas;
Harness::SpinBarrier& arena_barrier;
Harness::SpinBarrier& master_barrier;
ExecuteParallelFor(const int n_per_thread_, const int n_repetitions_,
std::vector<tbb::task_arena>& arenas_,
Harness::SpinBarrier& arena_barrier_, Harness::SpinBarrier& master_barrier_)
: n_per_thread(n_per_thread_), n_repetitions(n_repetitions_), arenas(arenas_),
arena_barrier(arena_barrier_), master_barrier(master_barrier_){ }
void operator()(int i) const {
for (int j = 0; j < n_repetitions; ++j) {
arenas[i].execute(TaskForArenaExecute(arena_barrier));
for(volatile int k = 0; k < n_per_thread; ++k){}
master_barrier.wait();
}
}
};

void TestArenaWorkersMigrationWithNumThreads(int n_threads = 0) {
if (n_threads == 0) {
n_threads = tbb::task_scheduler_init::default_num_threads();
}
const int max_n_arenas = 8;
int n_arenas = 2;
if(n_threads >= 16)
n_arenas = max_n_arenas;
else if (n_threads >= 8)
n_arenas = 4;
n_threads = n_arenas * (n_threads / n_arenas);
const int n_per_thread = 10000000;
const int n_repetitions = 100;
const int n_outer_repetitions = 20;
std::multiset<float> failure_ratio; 
tbb::global_control control(tbb::global_control::max_allowed_parallelism, n_threads - (n_arenas - 1));
Harness::SpinBarrier master_barrier(n_arenas);
Harness::SpinBarrier arena_barrier(n_threads);
MyObserver* observer[max_n_arenas];
std::vector<tbb::task_arena> arenas(n_arenas);
tbb::atomic<int> failure_counter;
tbb::atomic<int> counter;
tbb::enumerable_thread_specific<tbb::task_arena*> tls;
for (int i = 0; i < n_arenas; ++i) {
arenas[i].initialize(n_threads / n_arenas);
observer[i] = new MyObserver(arenas[i], tls, failure_counter, counter);
}
int ii = 0;
for (; ii < n_outer_repetitions; ++ii) {
failure_counter = 0;
counter = 0;
NativeParallelFor(n_arenas, ExecuteParallelFor(n_per_thread, n_repetitions,
arenas, arena_barrier, master_barrier));
failure_ratio.insert((counter != 0 ? float(failure_counter) / counter : 1.0f));
tls.clear();
if (ii > 1) {
std::multiset<float>::iterator it = failure_ratio.begin();
std::advance(it, failure_ratio.size() / 2);
if (*it < 0.02)
break;
}
}
for (int i = 0; i < n_arenas; ++i) {
delete observer[i];
}
std::multiset<float>::iterator it = failure_ratio.begin();
std::advance(it, failure_ratio.size() / 2);
if (*it > 0.05) {
REPORT("Warning: So many cases when threads join to different arenas.\n");
ASSERT(*it <= 0.3, "A lot of cases when threads join to different arenas.\n");
}
}

void TestArenaWorkersMigration() {
TestArenaWorkersMigrationWithNumThreads(4);
if (tbb::task_scheduler_init::default_num_threads() != 4) {
TestArenaWorkersMigrationWithNumThreads();
}
}

class CheckArenaNumThreads : public tbb::task {
public:
static Harness::SpinBarrier m_barrier;

CheckArenaNumThreads(int nt, int rm):  num_threads(nt), reserved_for_masters(rm) {
m_barrier.initialize(2);
}

tbb::task* execute() __TBB_override {
ASSERT( tbb::this_task_arena::max_concurrency() == num_threads, "Wrong concurrency of current arena" );
ASSERT( tbb::this_task_arena::current_thread_index() >= reserved_for_masters, "Thread shouldn't attach to master's slots" );
m_barrier.wait();
return NULL;
}

private:
const int num_threads;
const int reserved_for_masters;
};

Harness::SpinBarrier CheckArenaNumThreads::m_barrier;

class EnqueueTaskIntoTaskArena
{
public:
EnqueueTaskIntoTaskArena(tbb::task& t, tbb::task_arena& a) : my_task(t), my_arena(a) {}
void operator() ()
{
tbb::task::enqueue(my_task, my_arena);
}
private:
tbb::task& my_task;
tbb::task_arena& my_arena;
};

void TestTaskEnqueueInArena()
{   
int pp[8]={3, 4, 5, 7, 8, 11, 13, 17};
for(int i = 0; i < 8; ++i)
{
int p = pp[i];
int reserved_for_masters = p - 1;
tbb::task_arena a(p, reserved_for_masters);
a.initialize();
{
CheckArenaNumThreads& t = *new( tbb::task::allocate_root() ) CheckArenaNumThreads(p, reserved_for_masters);
tbb::task::enqueue(t, a);
CheckArenaNumThreads::m_barrier.wait();
a.debug_wait_until_empty();
}
{
CheckArenaNumThreads& t = *new( tbb::task::allocate_root() ) CheckArenaNumThreads(p, reserved_for_masters);
tbb::tbb_thread thr(EnqueueTaskIntoTaskArena(t, a));
CheckArenaNumThreads::m_barrier.wait();
a.debug_wait_until_empty();
thr.join();
}
}
}


int TestMain() {
#if __TBB_TASK_ISOLATION
TestIsolatedExecute();
#endif 
TestSmallStackSize();
TestDefaultWorkersLimit();
tbb::task_scheduler_init init_market_p_plus_one(MaxThread + 1);
TestConcurrentFunctionality();
TestArenaEntryConsistency();
TestAttach(MaxThread);
TestConstantFunctorRequirement();
TestDelegatedSpawnWait();
TestMultipleWaits();
TestMoveSemantics();
TestReturnValue();
TestArenaWorkersMigration();
TestTaskEnqueueInArena();
return Harness::Done;
}
