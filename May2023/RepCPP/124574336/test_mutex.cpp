

#include "harness_defs.h"
#include "tbb/spin_mutex.h"
#include "tbb/critical_section.h"
#include "tbb/spin_rw_mutex.h"
#include "tbb/queuing_rw_mutex.h"
#include "tbb/queuing_mutex.h"
#include "tbb/mutex.h"
#include "tbb/recursive_mutex.h"
#include "tbb/null_mutex.h"
#include "tbb/null_rw_mutex.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/tick_count.h"
#include "tbb/atomic.h"
#include "harness.h"
#include <cstdlib>
#include <cstdio>
#if _OPENMP
#include "test/OpenMP_Mutex.h"
#endif 
#include "tbb/tbb_profiling.h"

#ifndef TBB_TEST_LOW_WORKLOAD
#define TBB_TEST_LOW_WORKLOAD TBB_USE_THREADING_TOOLS
#endif


template<typename M>
struct Counter {
typedef M mutex_type;
M mutex;
volatile long value;
};

template<typename C>
struct AddOne: NoAssign {
C& counter;

void operator()( tbb::blocked_range<size_t>& range ) const {
for( size_t i=range.begin(); i!=range.end(); ++i ) {
if( i&1 ) {
typename C::mutex_type::scoped_lock lock(counter.mutex);
counter.value = counter.value+1;
lock.release();
} else {
typename C::mutex_type::scoped_lock lock;
lock.acquire(counter.mutex);
counter.value = counter.value+1;
}
}
}
AddOne( C& counter_ ) : counter(counter_) {}
};

template<typename M>
class TBB_MutexFromISO_Mutex {
M my_iso_mutex;
public:
typedef TBB_MutexFromISO_Mutex mutex_type;

class scoped_lock;
friend class scoped_lock;

class scoped_lock {
mutex_type* my_mutex;
public:
scoped_lock() : my_mutex(NULL) {}
scoped_lock( mutex_type& m ) : my_mutex(NULL) {
acquire(m);
}
scoped_lock( mutex_type& m, bool is_writer ) : my_mutex(NULL) {
acquire(m,is_writer);
}
void acquire( mutex_type& m ) {
m.my_iso_mutex.lock();
my_mutex = &m;
}
bool try_acquire( mutex_type& m ) {
if( m.my_iso_mutex.try_lock() ) {
my_mutex = &m;
return true;
} else {
return false;
}
}
void release() {
my_mutex->my_iso_mutex.unlock();
my_mutex = NULL;
}


void acquire( mutex_type& m, bool is_writer ) {
if( is_writer ) m.my_iso_mutex.lock();
else m.my_iso_mutex.lock_read();
my_mutex = &m;
}
bool try_acquire( mutex_type& m, bool is_writer ) {
if( is_writer ? m.my_iso_mutex.try_lock() : m.my_iso_mutex.try_lock_read() ) {
my_mutex = &m;
return true;
} else {
return false;
}
}
bool upgrade_to_writer() {
my_mutex->my_iso_mutex.unlock();
my_mutex->my_iso_mutex.lock();
return false;
}
bool downgrade_to_reader() {
my_mutex->my_iso_mutex.unlock();
my_mutex->my_iso_mutex.lock_read();
return false;
}
~scoped_lock() {
if( my_mutex )
release();
}
};

static const bool is_recursive_mutex = M::is_recursive_mutex;
static const bool is_rw_mutex = M::is_rw_mutex;
};

namespace tbb {
namespace profiling {
template<typename M>
void set_name( const TBB_MutexFromISO_Mutex<M>&, const char* ) {}
}
}


template<typename M>
void Test( const char * name ) {
REMARK("%s size == %d, time = ",name, sizeof(M));
Counter<M> counter;
counter.value = 0;
tbb::profiling::set_name(counter.mutex, name);
#if TBB_TEST_LOW_WORKLOAD
const int n = 10000;
#else
const int n = 100000;
#endif 
tbb::tick_count t0 = tbb::tick_count::now();
tbb::parallel_for(tbb::blocked_range<size_t>(0,n,n/10),AddOne<Counter<M> >(counter));
tbb::tick_count t1 = tbb::tick_count::now();
REMARK("%g usec\n",(t1-t0).seconds());
if( counter.value!=n )
REPORT("ERROR for %s: counter.value=%ld\n",name,counter.value);
}

template<typename M, size_t N>
struct Invariant {
typedef M mutex_type;
M mutex;
const char* mutex_name;
volatile long value[N];
Invariant( const char* mutex_name_ ) :
mutex_name(mutex_name_)
{
for( size_t k=0; k<N; ++k )
value[k] = 0;
tbb::profiling::set_name(mutex, mutex_name_);
}
~Invariant() {
}
void update() {
for( size_t k=0; k<N; ++k )
++value[k];
}
bool value_is( long expected_value ) const {
long tmp;
for( size_t k=0; k<N; ++k )
if( (tmp=value[k])!=expected_value ) {
REPORT("ERROR: %ld!=%ld\n", tmp, expected_value);
return false;
}
return true;
}
bool is_okay() {
return value_is( value[0] );
}
};

template<typename I>
struct TwiddleInvariant: NoAssign {
I& invariant;
TwiddleInvariant( I& invariant_ ) : invariant(invariant_) {}


void operator()( tbb::blocked_range<size_t>& range ) const {
for( size_t i=range.begin(); i!=range.end(); ++i ) {
const bool write = (i%8)==7;
bool okay = true;
bool lock_kept = true;
if( (i/8)&1 ) {
typename I::mutex_type::scoped_lock lock(invariant.mutex,write);
execute_aux(lock, i, write, okay, lock_kept);
lock.release();
} else {
typename I::mutex_type::scoped_lock lock;
lock.acquire(invariant.mutex,write);
execute_aux(lock, i, write, okay, lock_kept);
}
if( !okay ) {
REPORT( "ERROR for %s at %ld: %s %s %s %s\n",invariant.mutex_name, long(i),
write     ? "write,"                  : "read,",
write     ? (i%16==7?"downgrade,":"") : (i%8==3?"upgrade,":""),
lock_kept ? "lock kept,"              : "lock not kept,", 
(i/8)&1   ? "impl/expl"               : "expl/impl" );
}
}
}
private:
void execute_aux(typename I::mutex_type::scoped_lock & lock, const size_t i, const bool write, bool & okay, bool & lock_kept) const {
if( write ) {
long my_value = invariant.value[0];
invariant.update();
if( i%16==7 ) {
lock_kept = lock.downgrade_to_reader();
if( !lock_kept )
my_value = invariant.value[0] - 1;
okay = invariant.value_is(my_value+1);
}
} else {
okay = invariant.is_okay();
if( i%8==3 ) {
long my_value = invariant.value[0];
lock_kept = lock.upgrade_to_writer();
if( !lock_kept )
my_value = invariant.value[0];
invariant.update();
okay = invariant.value_is(my_value+1);
}
}
}
};


template<typename M>
void TestReaderWriterLock( const char * mutex_name ) {
REMARK( "%s readers & writers time = ", mutex_name );
Invariant<M,8> invariant(mutex_name);
#if TBB_TEST_LOW_WORKLOAD
const size_t n = 10000;
#else
const size_t n = 500000;
#endif 
tbb::tick_count t0 = tbb::tick_count::now();
tbb::parallel_for(tbb::blocked_range<size_t>(0,n,n/100),TwiddleInvariant<Invariant<M,8> >(invariant));
tbb::tick_count t1 = tbb::tick_count::now();
long expected_value = n/4;
if( !invariant.value_is(expected_value) )
REPORT("ERROR for %s: final invariant value is wrong\n",mutex_name);
REMARK( "%g usec\n", (t1-t0).seconds() );
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( push )
#pragma warning( disable: 4127 )
#endif


template<typename M>
void TestTryAcquireReader_OneThread( const char * mutex_name ) {
M tested_mutex;
typename M::scoped_lock lock1;
if( M::is_rw_mutex ) {
if( lock1.try_acquire(tested_mutex, false) )
lock1.release();
else
REPORT("ERROR for %s: try_acquire failed though it should not\n", mutex_name);
{
typename M::scoped_lock lock2(tested_mutex, false);   
if( lock1.try_acquire(tested_mutex) )                 
REPORT("ERROR for %s: try_acquire succeeded though it should not (1)\n", mutex_name);
lock2.release();                                      
lock2.acquire(tested_mutex, true);                    
if( lock1.try_acquire(tested_mutex, false) )          
REPORT("ERROR for %s: try_acquire succeeded though it should not (2)\n", mutex_name);
}
if( lock1.try_acquire(tested_mutex, false) )
lock1.release();
else
REPORT("ERROR for %s: try_acquire failed though it should not\n", mutex_name);
}
}


template<typename M>
void TestTryAcquire_OneThread( const char * mutex_name ) {
M tested_mutex;
typename M::scoped_lock lock1;
if( lock1.try_acquire(tested_mutex) )
lock1.release();
else
REPORT("ERROR for %s: try_acquire failed though it should not\n", mutex_name);
{
if( M::is_recursive_mutex ) {
typename M::scoped_lock lock2(tested_mutex);
if( lock1.try_acquire(tested_mutex) )
lock1.release();
else
REPORT("ERROR for %s: try_acquire on recursive lock failed though it should not\n", mutex_name);
} else {
typename M::scoped_lock lock2(tested_mutex);
if( lock1.try_acquire(tested_mutex) )
REPORT("ERROR for %s: try_acquire succeeded though it should not (3)\n", mutex_name);
}
}
if( lock1.try_acquire(tested_mutex) )
lock1.release();
else
REPORT("ERROR for %s: try_acquire failed though it should not\n", mutex_name);
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning( pop )
#endif

const int RecurN = 4;
int RecurArray[ RecurN ];
tbb::recursive_mutex RecurMutex[ RecurN ];

struct RecursiveAcquisition {

void Body( size_t x, int max_lock=-1, unsigned int mask=0 ) const
{
int i = (int) (x % RecurN);
bool first = (mask&1U<<i)==0;
if( first ) {
if( i<max_lock )
return;
max_lock = i;
}

if( (i&1)!=0 ) {
tbb::recursive_mutex::scoped_lock r_lock;
r_lock.acquire( RecurMutex[i] );
int a = RecurArray[i];
ASSERT( (a==0)==first, "should be either a==0 if it is the first time to acquire the lock or a!=0 otherwise" );
++RecurArray[i];
if( x )
Body( x/RecurN, max_lock, mask|1U<<i );
--RecurArray[i];
ASSERT( a==RecurArray[i], "a is not equal to RecurArray[i]" );

if( (i&2)!=0 ) r_lock.release();
} else {
tbb::recursive_mutex::scoped_lock r_lock( RecurMutex[i] );
int a = RecurArray[i];

ASSERT( (a==0)==first, "should be either a==0 if it is the first time to acquire the lock or a!=0 otherwise" );

++RecurArray[i];
if( x )
Body( x/RecurN, max_lock, mask|1U<<i );
--RecurArray[i];

ASSERT( a==RecurArray[i], "a is not equal to RecurArray[i]" );

if( (i&2)!=0 ) r_lock.release();
}
}

void operator()( const tbb::blocked_range<size_t> &r ) const
{
for( size_t x=r.begin(); x<r.end(); x++ ) {
Body( x );
}
}
};


template<typename M>
void TestRecursiveMutex( const char * mutex_name )
{
for ( int i = 0; i < RecurN; ++i ) {
tbb::profiling::set_name(RecurMutex[i], mutex_name);
}
tbb::tick_count t0 = tbb::tick_count::now();
tbb::parallel_for(tbb::blocked_range<size_t>(0,10000,500), RecursiveAcquisition());
tbb::tick_count t1 = tbb::tick_count::now();
REMARK( "%s recursive mutex time = %g usec\n", mutex_name, (t1-t0).seconds() );
}

template<typename C>
struct NullRecursive: NoAssign {
void recurse_till( size_t i, size_t till ) const {
if( i==till ) {
counter.value = counter.value+1;
return;
}
if( i&1 ) {
typename C::mutex_type::scoped_lock lock2(counter.mutex);
recurse_till( i+1, till );
lock2.release();
} else {
typename C::mutex_type::scoped_lock lock2;
lock2.acquire(counter.mutex);
recurse_till( i+1, till );
}
}

void operator()( tbb::blocked_range<size_t>& range ) const {
typename C::mutex_type::scoped_lock lock(counter.mutex);
recurse_till( range.begin(), range.end() );
}
NullRecursive( C& counter_ ) : counter(counter_) {
ASSERT( C::mutex_type::is_recursive_mutex, "Null mutex should be a recursive mutex." );
}
C& counter;
};

template<typename M>
struct NullUpgradeDowngrade: NoAssign {
void operator()( tbb::blocked_range<size_t>& range ) const {
typename M::scoped_lock lock2;
for( size_t i=range.begin(); i!=range.end(); ++i ) {
if( i&1 ) {
typename M::scoped_lock lock1(my_mutex, true) ;
if( lock1.downgrade_to_reader()==false )
REPORT("ERROR for %s: downgrade should always succeed\n", name);
} else {
lock2.acquire( my_mutex, false );
if( lock2.upgrade_to_writer()==false )
REPORT("ERROR for %s: upgrade should always succeed\n", name);
lock2.release();
}
}
}

NullUpgradeDowngrade( M& m_, const char* n_ ) : my_mutex(m_), name(n_) {}
M& my_mutex;
const char* name;
} ;

template<typename M>
void TestNullMutex( const char * name ) {
Counter<M> counter;
counter.value = 0;
const int n = 100;
REMARK("TestNullMutex<%s>",name);
{
tbb::parallel_for(tbb::blocked_range<size_t>(0,n,10),AddOne<Counter<M> >(counter));
}
counter.value = 0;
{
tbb::parallel_for(tbb::blocked_range<size_t>(0,n,10),NullRecursive<Counter<M> >(counter));
}
REMARK("\n");
}

template<typename M>
void TestNullRWMutex( const char * name ) {
REMARK("TestNullRWMutex<%s>",name);
const int n = 100;
M m;
tbb::parallel_for(tbb::blocked_range<size_t>(0,n,10),NullUpgradeDowngrade<M>(m, name));
REMARK("\n");
}

template<typename M>
void TestISO( const char * name ) {
typedef TBB_MutexFromISO_Mutex<M> tbb_from_iso;
Test<tbb_from_iso>( name );
}

template<typename M>
void TestTryAcquire_OneThreadISO( const char * name ) {
typedef TBB_MutexFromISO_Mutex<M> tbb_from_iso;
TestTryAcquire_OneThread<tbb_from_iso>( name );
}

template<typename M>
void TestReaderWriterLockISO( const char * name ) {
typedef TBB_MutexFromISO_Mutex<M> tbb_from_iso;
TestReaderWriterLock<tbb_from_iso>( name );
TestTryAcquireReader_OneThread<tbb_from_iso>( name );
}

template<typename M>
void TestRecursiveMutexISO( const char * name ) {
typedef TBB_MutexFromISO_Mutex<M> tbb_from_iso;
TestRecursiveMutex<tbb_from_iso>(name);
}

#include "harness_tsx.h"
#include "tbb/task_scheduler_init.h"

#if __TBB_TSX_TESTING_ENABLED_FOR_THIS_COMPILER

tbb::atomic<size_t> n_transactions_attempted;
template<typename C>
struct AddOne_CheckTransaction: NoAssign {
C& counter;

void operator()( tbb::blocked_range<size_t>& range ) const {
for( size_t i=range.begin(); i!=range.end(); ++i ) {
bool transaction_attempted = false;
{
typename C::mutex_type::scoped_lock lock(counter.mutex);
if( IsInsideTx() ) transaction_attempted = true;
counter.value = counter.value+1;
}
if( transaction_attempted ) ++n_transactions_attempted;
__TBB_Pause(i);
}
}
AddOne_CheckTransaction( C& counter_ ) : counter(counter_) {}
};


template<typename M>
void TestTransaction( const char * name )
{
Counter<M> counter;
#if TBB_TEST_LOW_WORKLOAD
const int n = 100;
#else
const int n = 1000;
#endif
REMARK("TestTransaction with %s: ",name);

n_transactions_attempted = 0;
tbb::tick_count start, stop;
for( int i=0; i<5 && n_transactions_attempted==0; ++i ) {
counter.value = 0;
start = tbb::tick_count::now();
tbb::parallel_for(tbb::blocked_range<size_t>(0,n,2),AddOne_CheckTransaction<Counter<M> >(counter));
stop = tbb::tick_count::now();
if( counter.value!=n ) {
REPORT("ERROR for %s: counter.value=%ld\n",name,counter.value);
break;
}
}

if( n_transactions_attempted==0 )
REPORT( "ERROR: transactions were never attempted\n" );
else
REMARK("%d successful transactions in %6.6f seconds\n", (int)n_transactions_attempted, (stop - start).seconds());
}
#endif  

template<typename M>
class RWStateMultipleChangeBody {
M& my_mutex;
public:
RWStateMultipleChangeBody(M& m) : my_mutex(m) {}

void operator()(const tbb::blocked_range<size_t>& r) const {
typename M::scoped_lock l(my_mutex, false);
for(size_t i = r.begin(); i != r.end(); ++i) {
ASSERT(l.downgrade_to_reader(), "Downgrade must succeed for read lock");
}
l.upgrade_to_writer();
for(size_t i = r.begin(); i != r.end(); ++i) {
ASSERT(l.upgrade_to_writer(), "Upgrade must succeed for write lock");
}
}
};

template<typename M>
void TestRWStateMultipleChange() {
ASSERT(M::is_rw_mutex, "Incorrect mutex type");
size_t n = 10000;
M mutex;
RWStateMultipleChangeBody<M> body(mutex);
tbb::parallel_for(tbb::blocked_range<size_t>(0, n, n/10), body);
}

int TestMain () {
for( int p=MinThread; p<=MaxThread; ++p ) {
tbb::task_scheduler_init init( p );
REMARK( "testing with %d workers\n", static_cast<int>(p) );
#if TBB_TEST_LOW_WORKLOAD
const int n = 1;
#else
const int n = 3;
#endif
for( int i=0; i<n; ++i ) {
TestNullMutex<tbb::null_mutex>( "Null Mutex" );
TestNullMutex<tbb::null_rw_mutex>( "Null RW Mutex" );
TestNullRWMutex<tbb::null_rw_mutex>( "Null RW Mutex" );
Test<tbb::spin_mutex>( "Spin Mutex" );
Test<tbb::speculative_spin_mutex>( "Spin Mutex/speculative" );
#if _OPENMP
Test<OpenMP_Mutex>( "OpenMP_Mutex" );
#endif 
Test<tbb::queuing_mutex>( "Queuing Mutex" );
Test<tbb::mutex>( "Wrapper Mutex" );
Test<tbb::recursive_mutex>( "Recursive Mutex" );
Test<tbb::queuing_rw_mutex>( "Queuing RW Mutex" );
Test<tbb::spin_rw_mutex>( "Spin RW Mutex" );
Test<tbb::speculative_spin_rw_mutex>( "Spin RW Mutex/speculative" );

TestTryAcquire_OneThread<tbb::spin_mutex>("Spin Mutex");
TestTryAcquire_OneThread<tbb::speculative_spin_mutex>("Spin Mutex/speculative");
TestTryAcquire_OneThread<tbb::queuing_mutex>("Queuing Mutex");
#if USE_PTHREAD
TestTryAcquire_OneThread<tbb::mutex>("Wrapper Mutex");
#endif 
TestTryAcquire_OneThread<tbb::recursive_mutex>( "Recursive Mutex" );
TestTryAcquire_OneThread<tbb::spin_rw_mutex>("Spin RW Mutex"); 
TestTryAcquire_OneThread<tbb::speculative_spin_rw_mutex>("Spin RW Mutex/speculative"); 
TestTryAcquire_OneThread<tbb::queuing_rw_mutex>("Queuing RW Mutex"); 

TestTryAcquireReader_OneThread<tbb::spin_rw_mutex>("Spin RW Mutex");
TestTryAcquireReader_OneThread<tbb::speculative_spin_rw_mutex>("Spin RW Mutex/speculative");
TestTryAcquireReader_OneThread<tbb::queuing_rw_mutex>("Queuing RW Mutex");

TestReaderWriterLock<tbb::queuing_rw_mutex>( "Queuing RW Mutex" );
TestReaderWriterLock<tbb::spin_rw_mutex>( "Spin RW Mutex" );
TestReaderWriterLock<tbb::speculative_spin_rw_mutex>( "Spin RW Mutex/speculative" );

TestRecursiveMutex<tbb::recursive_mutex>( "Recursive Mutex" );

TestISO<tbb::spin_mutex>( "ISO Spin Mutex" );
TestISO<tbb::mutex>( "ISO Mutex" );
TestISO<tbb::spin_rw_mutex>( "ISO Spin RW Mutex" );
TestISO<tbb::recursive_mutex>( "ISO Recursive Mutex" );
TestISO<tbb::critical_section>( "ISO Critical Section" );
TestTryAcquire_OneThreadISO<tbb::spin_mutex>( "ISO Spin Mutex" );
#if USE_PTHREAD
TestTryAcquire_OneThreadISO<tbb::mutex>( "ISO Mutex" );
#endif 
TestTryAcquire_OneThreadISO<tbb::spin_rw_mutex>( "ISO Spin RW Mutex" );
TestTryAcquire_OneThreadISO<tbb::recursive_mutex>( "ISO Recursive Mutex" );
TestTryAcquire_OneThreadISO<tbb::critical_section>( "ISO Critical Section" );
TestReaderWriterLockISO<tbb::spin_rw_mutex>( "ISO Spin RW Mutex" );
TestRecursiveMutexISO<tbb::recursive_mutex>( "ISO Recursive Mutex" );

TestRWStateMultipleChange<tbb::spin_rw_mutex>();
TestRWStateMultipleChange<tbb::speculative_spin_rw_mutex>();
TestRWStateMultipleChange<tbb::queuing_rw_mutex>();
}
}

#if __TBB_TSX_TESTING_ENABLED_FOR_THIS_COMPILER
if( have_TSX() ) {
tbb::task_scheduler_init init( MaxThread );
TestTransaction<tbb::speculative_spin_mutex>( "Spin Mutex/speculative" );
TestTransaction<tbb::speculative_spin_rw_mutex>( "Spin RW Mutex/speculative" );
}
else {
REMARK("Hardware transactions not supported\n");
}
#endif
return Harness::Done;
}
