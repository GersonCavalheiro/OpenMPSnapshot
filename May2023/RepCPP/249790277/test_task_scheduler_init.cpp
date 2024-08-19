

#if !__TBB_CPF_BUILD
#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#endif

#include "tbb/task_scheduler_init.h"
#include <cstdlib>
#include "harness_assert.h"

#include <cstdio>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#if TBB_USE_EXCEPTIONS
#include <stdexcept>
#endif

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#if _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4702)
#endif
#include "tbb/parallel_for.h"
#if _MSC_VER
#pragma warning (pop)
#endif

#include "harness_concurrency_tracker.h"
#include "harness_task.h"
#include "harness.h"

const int DefaultThreads = tbb::task_scheduler_init::default_num_threads();

namespace tbb { namespace internal {
size_t __TBB_EXPORTED_FUNC get_initial_auto_partitioner_divisor();
}}

int ArenaConcurrency() {
return int(tbb::internal::get_initial_auto_partitioner_divisor()/4); 
}

bool test_mandatory_parallelism = true;


void InitializeAndTerminate( int maxthread ) {
__TBB_TRY {
for( int i=0; i<256; ++i ) {
int threads = (std::rand() % maxthread) + 1;
switch( i&3 ) {
default: {
tbb::task_scheduler_init init( threads );
ASSERT(init.is_active(), NULL);
ASSERT(ArenaConcurrency()==(threads==1?2:threads), NULL);
if (test_mandatory_parallelism)
Harness::ExactConcurrencyLevel::check(threads, Harness::ExactConcurrencyLevel::Serialize);
if(i&0x20) tbb::task::enqueue( (*new( tbb::task::allocate_root() ) TaskGenerator(2,6)) ); 
break;
}
case 0: {
tbb::task_scheduler_init init;
ASSERT(init.is_active(), NULL);
ASSERT(ArenaConcurrency()==(DefaultThreads==1?2:init.default_num_threads()), NULL);
if (test_mandatory_parallelism)
Harness::ExactConcurrencyLevel::check(init.default_num_threads(), Harness::ExactConcurrencyLevel::Serialize);
if(i&0x40) tbb::task::enqueue( (*new( tbb::task::allocate_root() ) TaskGenerator(3,5)) ); 
break;
}
case 1: {
tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
ASSERT(!init.is_active(), "init should not be active; initialization was deferred");
init.initialize( threads );
ASSERT(init.is_active(), NULL);
ASSERT(ArenaConcurrency()==(threads==1?2:threads), NULL);
if (test_mandatory_parallelism)
Harness::ExactConcurrencyLevel::check(threads, Harness::ExactConcurrencyLevel::Serialize);
init.terminate();
ASSERT(!init.is_active(), "init should not be active; it was terminated");
break;
}
case 2: {
tbb::task_scheduler_init init( tbb::task_scheduler_init::automatic );
ASSERT(init.is_active(), NULL);
ASSERT(ArenaConcurrency()==(DefaultThreads==1?2:init.default_num_threads()), NULL);
if (test_mandatory_parallelism)
Harness::ExactConcurrencyLevel::check(init.default_num_threads(), Harness::ExactConcurrencyLevel::Serialize);
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

#if _WIN64
namespace std {      
using ::srand;
}
#endif 

struct ThreadedInit {
void operator()( int ) const {
InitializeAndTerminate(MaxThread);
}
};

#if _MSC_VER
#include "tbb/machine/windows_api.h"
#include <tchar.h>
#endif 


void AssertExplicitInitIsNotSupplanted () {
tbb::task_scheduler_init init(1);

Harness::ExactConcurrencyLevel::check(1);
}

struct TestNoWorkerSurplusRun {
void operator() (int) const {
const unsigned THREADS = tbb::tbb_thread::hardware_concurrency()*2/3;
for (int j=0; j<10; j++) {
tbb::task_scheduler_init t(THREADS);
Harness::ExactConcurrencyLevel::Combinable unique;

for (int i=0; i<50; i++)
Harness::ExactConcurrencyLevel::checkLessOrEqual(THREADS, &unique);
}
}
};

void TestNoWorkerSurplus () {
NativeParallelFor( 1, TestNoWorkerSurplusRun() );
}

#if TBB_PREVIEW_WAITING_FOR_WORKERS
#include "tbb/task_group.h"
#include "tbb/task_arena.h"

namespace TestBlockingTerminateNS {
struct EmptyBody {
void operator()() const {}
void operator()( int ) const {}
};

struct TestAutoInitBody {
void operator()( int ) const {
tbb::parallel_for( 0, 100, EmptyBody() );
}
};

static tbb::atomic<int> gSeed;
static tbb::atomic<int> gNumSuccesses;

class TestMultpleWaitBody {
bool myAutoInit;
public:
TestMultpleWaitBody( bool autoInit = false ) : myAutoInit( autoInit ) {}
void operator()( int ) const {
tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
if ( !myAutoInit )
init.initialize( tbb::task_scheduler_init::automatic );
Harness::FastRandom rnd( ++gSeed );
const int numCases = myAutoInit ? 4 : 6;
switch ( rnd.get() % numCases ) {
case 0: {
tbb::task_arena a;
a.enqueue( EmptyBody() );
break;
}
case 1: {
tbb::task_group tg;
tg.run( EmptyBody() );
tg.wait();
break;
}
case 2:
tbb::parallel_for( 0, 100, EmptyBody() );
break;
case 3:

break;
case 4:
NativeParallelFor( rnd.get() % 5 + 1, TestMultpleWaitBody( true ) );
break;
case 5:
{
tbb::task_scheduler_init init2;
bool res = init2.blocking_terminate( std::nothrow );
ASSERT( !res, NULL );
}
break;
}
if ( !myAutoInit && init.blocking_terminate( std::nothrow ) )
++gNumSuccesses;
}
};

void TestMultpleWait() {
const int minThreads = 1;
const int maxThreads = 16;
const int numRepeats = 5;
gSeed = tbb::task_scheduler_init::default_num_threads();
for ( int repeats = 0; repeats<numRepeats; ++repeats ) {
for ( int threads = minThreads; threads<maxThreads; ++threads ) {
gNumSuccesses = 0;
NativeParallelFor( threads, TestMultpleWaitBody() );
ASSERT( gNumSuccesses > 0, "At least one blocking terminate must return 'true'" );
}
}
}

#if TBB_USE_EXCEPTIONS
template <typename F>
void TestException( F &f ) {
Harness::suppress_unused_warning( f );
bool caught = false;
try {
f();
ASSERT( false, NULL );
}
catch ( const std::runtime_error& ) {
caught = true;
}
#if TBB_USE_CAPTURED_EXCEPTION
catch ( const tbb::captured_exception& ) {
caught = true;
}
#endif
catch ( ... ) {
ASSERT( false, NULL );
}
ASSERT( caught, NULL );
}

class ExceptionTest1 {
tbb::task_scheduler_init tsi1;
int myIndex;
public:
ExceptionTest1( int index ) : myIndex( index ) {}

void operator()() {
tbb::task_scheduler_init tsi2;
(myIndex == 0 ? tsi1 : tsi2).blocking_terminate();
ASSERT( false, "Blocking terminate did not throw the exception" );
}
};

struct ExceptionTest2 {
class Body {
Harness::SpinBarrier& myBarrier;
public:
Body( Harness::SpinBarrier& barrier ) : myBarrier( barrier ) {}
void operator()( int ) const {
myBarrier.wait();
tbb::task_scheduler_init init;
init.blocking_terminate();
ASSERT( false, "Blocking terminate did not throw the exception inside the parallel region" );
}
};
void operator()() {
const int numThreads = 4;
tbb::task_scheduler_init init( numThreads );
Harness::SpinBarrier barrier( numThreads );
tbb::parallel_for( 0, numThreads, Body( barrier ) );
ASSERT( false, "Parallel loop did not throw the exception" );
}
};
#endif 

void TestExceptions() {
for ( int i = 0; i<2; ++i ) {
tbb::task_scheduler_init tsi[2];
bool res1 = tsi[i].blocking_terminate( std::nothrow );
ASSERT( !res1, NULL );
bool res2 = tsi[1-i].blocking_terminate( std::nothrow );
ASSERT( res2, NULL );
}
#if TBB_USE_EXCEPTIONS
ExceptionTest1 Test1(0), Test2(1);
TestException( Test1 );
TestException( Test2 );
ExceptionTest2 Test3;
TestException( Test3 );
#endif
}
}

void TestBlockingTerminate() {
TestBlockingTerminateNS::TestExceptions();
TestBlockingTerminateNS::TestMultpleWait();
}
#endif 

int TestMain () {
#if _MSC_VER && !__TBB_NO_IMPLICIT_LINKAGE && !defined(__TBB_LIB_NAME)
#ifdef _DEBUG
ASSERT(!GetModuleHandle(_T("tbb.dll")) && GetModuleHandle(_T("tbb_debug.dll")),
"test linked with wrong (non-debug) tbb library");
#else
ASSERT(!GetModuleHandle(_T("tbb_debug.dll")) && GetModuleHandle(_T("tbb.dll")),
"test linked with wrong (debug) tbb library");
#endif
#endif 
std::srand(2);
REMARK("testing master thread\n");
int threads = DefaultThreads*2;
{   
tbb::task_scheduler_init init( threads );
if( !Harness::ExactConcurrencyLevel::isEqual( threads ) ) {
threads = DefaultThreads;
if( MaxThread > DefaultThreads )
MaxThread = DefaultThreads;
#if RML_USE_WCRM
REPORT("Known issue: shared RML for ConcRT does not support oversubscription\n");
test_mandatory_parallelism = false; 
#else
REPORT("Known issue: machine is heavy loaded or shared RML which does not support oversubscription is loaded\n");
#endif
}
}
InitializeAndTerminate( threads ); 
for( int p=MinThread; p<=MaxThread; ++p ) {
REMARK("testing with %d threads\n", p );
tbb::task_scheduler_init init( tbb::task_scheduler_init::deferred );
if( MaxThread > DefaultThreads ) init.initialize( MaxThread );
NativeParallelFor( p, ThreadedInit() );
}
AssertExplicitInitIsNotSupplanted();
#if TBB_PREVIEW_WAITING_FOR_WORKERS
TestBlockingTerminate();
#endif
return Harness::Done;
}
