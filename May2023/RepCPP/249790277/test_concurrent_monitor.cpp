

#define HARNESS_DEFAULT_MIN_THREADS 6
#define HARNESS_DEFAULT_MAX_THREADS 8

#include "tbb/concurrent_monitor.h"
#include "tbb/atomic.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "harness.h"
#if _WIN32||_WIN64
#include "tbb/dynamic_link.cpp"
#endif

#include "tbb/semaphore.cpp"
#include "tbb/concurrent_monitor.cpp"

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (disable: 4127)
#endif

using namespace tbb;

class QueuingMutex {
public:
QueuingMutex() { q_tail = NULL; }

class ScopedLock: internal::no_copy {
void Initialize() { mutex = NULL; }
public:
ScopedLock() {Initialize();}
ScopedLock( QueuingMutex& m, size_t test_mode ) { Initialize(); Acquire(m,test_mode); }
~ScopedLock() { if( mutex ) Release(); }
void Acquire( QueuingMutex& m, size_t test_mode );
void Release();
void SleepPerhaps();

private:
QueuingMutex* mutex;
ScopedLock* next;
uintptr_t going;
internal::concurrent_monitor::thread_context thr_ctx;
};

friend class ScopedLock;
private:
atomic<ScopedLock*> q_tail;
internal::concurrent_monitor waitq;
};

struct PredicateEq {
uintptr_t p;
PredicateEq( uintptr_t p_ ) : p(p_) {}
bool operator() ( uintptr_t v ) const {return p==v;}
};

struct QueuingMutex_Context {
const QueuingMutex::ScopedLock* lck;
QueuingMutex_Context( QueuingMutex::ScopedLock* l_ ) : lck(l_) {}
uintptr_t operator()() { return uintptr_t(lck); }
};

struct QueuingMutex_Until : NoAssign {
uintptr_t& flag;
QueuingMutex_Until( uintptr_t& f_ ) : flag(f_) {}
bool operator()() { return flag!=0ul; }
};

void QueuingMutex::ScopedLock::Acquire( QueuingMutex& m, size_t test_mode )
{
mutex = &m;
next  = NULL;
going = 0;

ScopedLock* pred = m.q_tail.fetch_and_store<tbb::release>(this);
if( pred ) {
#if TBB_USE_ASSERT
__TBB_control_consistency_helper(); 
ASSERT( !pred->next, "the predecessor has another successor!");
#endif
pred->next = this;
for( int i=0; i<16; ++i ) {
if( going!=0ul ) break;
__TBB_Yield();
}
int x = int( test_mode%3 );
switch( x ) {
case 0:
mutex->waitq.wait( QueuingMutex_Until(going), QueuingMutex_Context(this) );
break;
#if __TBB_LAMBDAS_PRESENT
case 1:
mutex->waitq.wait( [&](){ return going!=0ul; }, [=]() { return (uintptr_t)this; } );
break;
#endif
default:
SleepPerhaps();
break;
}
}

__TBB_control_consistency_helper(); 
}

void QueuingMutex::ScopedLock::Release( )
{
if( !next ) {
if( this == mutex->q_tail.compare_and_swap<tbb::release>(NULL, this) ) {
goto done;
}
spin_wait_while_eq( next, (ScopedLock*)0 );
}
__TBB_store_with_release(next->going, 1);
mutex->waitq.notify( PredicateEq(uintptr_t(next)) );
done:
Initialize();
}

void QueuingMutex::ScopedLock::SleepPerhaps()
{
bool slept = false;
internal::concurrent_monitor& mq = mutex->waitq;
mq.prepare_wait( thr_ctx, uintptr_t(this) );
while( going==0ul ) {
if( (slept=mq.commit_wait( thr_ctx ))==true && going!=0ul )
break;
slept = false;
mq.prepare_wait( thr_ctx, uintptr_t(this) );
}
if( !slept )
mq.cancel_wait( thr_ctx );
}

class SpinMutex {
public:
SpinMutex() : toggle(false) { flag = 0; }

class ScopedLock: internal::no_copy {
void Initialize() { mutex = NULL; }
public:
ScopedLock() {Initialize();}
ScopedLock( SpinMutex& m, size_t test_mode ) { Initialize(); Acquire(m,test_mode); }
~ScopedLock() { if( mutex ) Release(); }
void Acquire( SpinMutex& m, size_t test_mode );
void Release();
void SleepPerhaps();

private:
SpinMutex* mutex;
internal::concurrent_monitor::thread_context thr_ctx;
};

friend class ScopedLock;
friend struct SpinMutex_Until;
private:
tbb::atomic<unsigned> flag;
bool toggle;
internal::concurrent_monitor waitq;
};

struct SpinMutex_Context {
const SpinMutex::ScopedLock* lck;
SpinMutex_Context( SpinMutex::ScopedLock* l_ ) : lck(l_) {}
uintptr_t operator()() { return uintptr_t(lck); }
};

struct SpinMutex_Until {
const SpinMutex* mtx;
SpinMutex_Until( SpinMutex* m_ ) : mtx(m_) {}
bool operator()() { return mtx->flag==0; }
};

void SpinMutex::ScopedLock::Acquire( SpinMutex& m, size_t test_mode )
{
mutex = &m;
retry:
if( m.flag.compare_and_swap( 1, 0 )!=0 ) {
int x = int( test_mode%3 );
switch( x ) {
case 0:
mutex->waitq.wait( SpinMutex_Until(mutex), SpinMutex_Context(this) );
break;
#if __TBB_LAMBDAS_PRESENT
case 1:
mutex->waitq.wait( [&](){ return mutex->flag==0; }, [=]() { return (uintptr_t)this; } );
break;
#endif
default:
SleepPerhaps();
break;
}
goto retry;
}
}

void SpinMutex::ScopedLock::Release()
{
bool old_toggle = mutex->toggle;
mutex->toggle = !mutex->toggle;
mutex->flag = 0;
if( old_toggle )
mutex->waitq.notify_one();
else
mutex->waitq.notify_all();
}

void SpinMutex::ScopedLock::SleepPerhaps()
{
bool slept = false;
internal::concurrent_monitor& mq = mutex->waitq;
mq.prepare_wait( thr_ctx, uintptr_t(this) );
while( mutex->flag ) {
if( (slept=mq.commit_wait( thr_ctx ))==true )
break;
mq.prepare_wait( thr_ctx, uintptr_t(this) );
}
if( !slept )
mq.cancel_wait( thr_ctx );
}

template<typename M>
struct Counter {
typedef M mutex_type;
M mutex;
long value;
};

template<typename C, int D>
struct AddOne: NoAssign {
C& counter;

void operator()( tbb::blocked_range<size_t>& range ) const {
for( size_t i=range.begin(); i!=range.end(); ++i ) {
typename C::mutex_type::ScopedLock lock(counter.mutex, i);
counter.value = counter.value+1;
if( D>0 )
for( int j=0; j<D; ++j ) __TBB_Yield();
}
}
AddOne( C& counter_ ) : counter(counter_) {}
};

template<typename M,int R, int D>
void Test( int p ) {
Counter<M> counter;
counter.value = 0;
const int n = R;
tbb::task_scheduler_init init(p);
tbb::parallel_for(tbb::blocked_range<size_t>(0,n,n/10),AddOne<Counter<M>,D>(counter));
if( counter.value!=n )
REPORT("ERROR : counter.value=%ld (instead of %ld)\n",counter.value,n);
}

#if TBB_USE_EXCEPTIONS
#define NTHRS_USED_IN_DESTRUCTOR_TEST 8

atomic<size_t> n_sleepers;

#if defined(_MSC_VER) && defined(_Wp64)
#pragma warning (disable: 4244 4267)
#endif

struct AllButOneSleep : NoAssign {
internal::concurrent_monitor*& mon;
static const size_t VLN = 1024*1024;
void operator()( int i ) const {
internal::concurrent_monitor::thread_context thr_ctx;

if( i==0 ) {
size_t n_expected_sleepers = NTHRS_USED_IN_DESTRUCTOR_TEST-1;
while( n_sleepers<n_expected_sleepers )
__TBB_Yield();
while( n_sleepers.compare_and_swap( VLN+NTHRS_USED_IN_DESTRUCTOR_TEST, n_expected_sleepers )!=n_expected_sleepers )
__TBB_Yield();

for( int j=0; j<100; ++j )
Harness::Sleep( 1 );
delete mon;
mon = NULL;
} else {
mon->prepare_wait( thr_ctx, uintptr_t(this) );
while( n_sleepers<VLN ) {
try {
++n_sleepers;
mon->commit_wait( thr_ctx );
if( --n_sleepers>VLN )
break;
} catch( tbb::user_abort& ) {
break;
}
mon->prepare_wait( thr_ctx, uintptr_t(this) );
}
}
}
AllButOneSleep( internal::concurrent_monitor*& m_ ) : mon(m_) {}
};
#endif 

void TestDestructor() {
#if TBB_USE_EXCEPTIONS
tbb::task_scheduler_init init(NTHRS_USED_IN_DESTRUCTOR_TEST);
internal::concurrent_monitor* my_mon = new internal::concurrent_monitor;
REMARK( "testing the destructor\n" );
n_sleepers = 0;
NativeParallelFor(NTHRS_USED_IN_DESTRUCTOR_TEST,AllButOneSleep(my_mon));
ASSERT( my_mon==NULL, "" );
#endif 
}

int TestMain () {
for( int p=MinThread; p<=MaxThread; ++p ) {
REMARK( "testing with %d workers\n", static_cast<int>(p) );
Test<QueuingMutex,100000,0>( p );
Test<QueuingMutex,1000,10000>( p );
Test<SpinMutex,100000,0>( p );
Test<SpinMutex,1000,10000>( p );
REMARK( "calling destructor for task_scheduler_init\n" );
}
TestDestructor();
return Harness::Done;
}
