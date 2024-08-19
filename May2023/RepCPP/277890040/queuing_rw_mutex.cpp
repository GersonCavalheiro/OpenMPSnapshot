



#include "tbb/queuing_rw_mutex.h"
#include "tbb/tbb_machine.h"
#include "tbb/tbb_stddef.h"
#include "tbb/tbb_machine.h"
#include "itt_notify.h"


namespace tbb {

using namespace internal;

enum state_t_flags {
STATE_NONE                   = 0,
STATE_WRITER                 = 1<<0,
STATE_READER                 = 1<<1,
STATE_READER_UNBLOCKNEXT     = 1<<2,
STATE_ACTIVEREADER           = 1<<3,
STATE_UPGRADE_REQUESTED      = 1<<4,
STATE_UPGRADE_WAITING        = 1<<5,
STATE_UPGRADE_LOSER          = 1<<6,
STATE_COMBINED_WAITINGREADER = STATE_READER | STATE_READER_UNBLOCKNEXT,
STATE_COMBINED_READER        = STATE_COMBINED_WAITINGREADER | STATE_ACTIVEREADER,
STATE_COMBINED_UPGRADING     = STATE_UPGRADE_WAITING | STATE_UPGRADE_LOSER
};

const unsigned char RELEASED = 0;
const unsigned char ACQUIRED = 1;

inline bool queuing_rw_mutex::scoped_lock::try_acquire_internal_lock()
{
return as_atomic(my_internal_lock).compare_and_swap<tbb::acquire>(ACQUIRED,RELEASED) == RELEASED;
}

inline void queuing_rw_mutex::scoped_lock::acquire_internal_lock()
{
while( !try_acquire_internal_lock() ) {
__TBB_Pause(1);
}
}

inline void queuing_rw_mutex::scoped_lock::release_internal_lock()
{
__TBB_store_with_release(my_internal_lock,RELEASED);
}

inline void queuing_rw_mutex::scoped_lock::wait_for_release_of_internal_lock()
{
spin_wait_until_eq(my_internal_lock, RELEASED);
}

inline void queuing_rw_mutex::scoped_lock::unblock_or_wait_on_internal_lock( uintptr_t flag ) {
if( flag )
wait_for_release_of_internal_lock();
else
release_internal_lock();
}

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4311 4312)
#endif

template<typename T>
class tricky_atomic_pointer: no_copy {
public:
typedef typename atomic_selector<sizeof(T*)>::word word;

template<memory_semantics M>
static T* fetch_and_add( T* volatile * location, word addend ) {
return reinterpret_cast<T*>( atomic_traits<sizeof(T*),M>::fetch_and_add(location, addend) );
}
template<memory_semantics M>
static T* fetch_and_store( T* volatile * location, T* value ) {
return reinterpret_cast<T*>( atomic_traits<sizeof(T*),M>::fetch_and_store(location, reinterpret_cast<word>(value)) );
}
template<memory_semantics M>
static T* compare_and_swap( T* volatile * location, T* value, T* comparand ) {
return reinterpret_cast<T*>(
atomic_traits<sizeof(T*),M>::compare_and_swap(location, reinterpret_cast<word>(value),
reinterpret_cast<word>(comparand))
);
}

T* & ref;
tricky_atomic_pointer( T*& original ) : ref(original) {};
tricky_atomic_pointer( T* volatile & original ) : ref(original) {};
T* operator&( word operand2 ) const {
return reinterpret_cast<T*>( reinterpret_cast<word>(ref) & operand2 );
}
T* operator|( word operand2 ) const {
return reinterpret_cast<T*>( reinterpret_cast<word>(ref) | operand2 );
}
};

typedef tricky_atomic_pointer<queuing_rw_mutex::scoped_lock> tricky_pointer;

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif

static const tricky_pointer::word FLAG = 0x1;

inline
uintptr_t get_flag( queuing_rw_mutex::scoped_lock* ptr ) {
return uintptr_t(ptr) & FLAG;
}


void queuing_rw_mutex::scoped_lock::acquire( queuing_rw_mutex& m, bool write )
{
__TBB_ASSERT( !my_mutex, "scoped_lock is already holding a mutex");

my_mutex = &m;
__TBB_store_relaxed(my_prev , (scoped_lock*)0);
__TBB_store_relaxed(my_next , (scoped_lock*)0);
__TBB_store_relaxed(my_going, 0);
my_state = state_t(write ? STATE_WRITER : STATE_READER);
my_internal_lock = RELEASED;

queuing_rw_mutex::scoped_lock* pred = m.q_tail.fetch_and_store<tbb::release>(this);

if( write ) {       

if( pred ) {
ITT_NOTIFY(sync_prepare, my_mutex);
pred = tricky_pointer(pred) & ~FLAG;
__TBB_ASSERT( !( uintptr_t(pred) & FLAG ), "use of corrupted pointer!" );
#if TBB_USE_ASSERT
__TBB_control_consistency_helper(); 
__TBB_ASSERT( !__TBB_load_relaxed(pred->my_next), "the predecessor has another successor!");
#endif
__TBB_store_with_release(pred->my_next,this);
spin_wait_until_eq(my_going, 1);
}

} else {            
#if DO_ITT_NOTIFY
bool sync_prepare_done = false;
#endif
if( pred ) {
unsigned short pred_state;
__TBB_ASSERT( !__TBB_load_relaxed(my_prev), "the predecessor is already set" );
if( uintptr_t(pred) & FLAG ) {

pred_state = STATE_UPGRADE_WAITING;
pred = tricky_pointer(pred) & ~FLAG;
} else {
pred_state = pred->my_state.compare_and_swap<tbb::acquire>(STATE_READER_UNBLOCKNEXT, STATE_READER);
}
__TBB_store_relaxed(my_prev, pred);
__TBB_ASSERT( !( uintptr_t(pred) & FLAG ), "use of corrupted pointer!" );
#if TBB_USE_ASSERT
__TBB_control_consistency_helper(); 
__TBB_ASSERT( !__TBB_load_relaxed(pred->my_next), "the predecessor has another successor!");
#endif
__TBB_store_with_release(pred->my_next,this);
if( pred_state != STATE_ACTIVEREADER ) {
#if DO_ITT_NOTIFY
sync_prepare_done = true;
ITT_NOTIFY(sync_prepare, my_mutex);
#endif
spin_wait_until_eq(my_going, 1);
}
}

unsigned short old_state = my_state.compare_and_swap<tbb::acquire>(STATE_ACTIVEREADER, STATE_READER);
if( old_state!=STATE_READER ) {
#if DO_ITT_NOTIFY
if( !sync_prepare_done )
ITT_NOTIFY(sync_prepare, my_mutex);
#endif
__TBB_ASSERT( my_state==STATE_READER_UNBLOCKNEXT, "unexpected state" );
spin_wait_while_eq(my_next, (scoped_lock*)NULL);

my_state = STATE_ACTIVEREADER;
__TBB_store_with_release(my_next->my_going,1);
}
}

ITT_NOTIFY(sync_acquired, my_mutex);

__TBB_load_with_acquire(my_going);
}

bool queuing_rw_mutex::scoped_lock::try_acquire( queuing_rw_mutex& m, bool write )
{
__TBB_ASSERT( !my_mutex, "scoped_lock is already holding a mutex");

if( load<relaxed>(m.q_tail) )
return false; 

__TBB_store_relaxed(my_prev, (scoped_lock*)0);
__TBB_store_relaxed(my_next, (scoped_lock*)0);
__TBB_store_relaxed(my_going, 0); 
my_state = state_t(write ? STATE_WRITER : STATE_ACTIVEREADER);
my_internal_lock = RELEASED;

if( m.q_tail.compare_and_swap<tbb::release>(this, NULL) )
return false; 
__TBB_load_with_acquire(my_going);
my_mutex = &m;
ITT_NOTIFY(sync_acquired, my_mutex);
return true;
}

void queuing_rw_mutex::scoped_lock::release( )
{
__TBB_ASSERT(my_mutex!=NULL, "no lock acquired");

ITT_NOTIFY(sync_releasing, my_mutex);

if( my_state == STATE_WRITER ) { 

scoped_lock* n = __TBB_load_with_acquire(my_next);
if( !n ) {
if( this == my_mutex->q_tail.compare_and_swap<tbb::release>(NULL, this) ) {
goto done;
}
spin_wait_while_eq( my_next, (scoped_lock*)NULL );
n = __TBB_load_with_acquire(my_next);
}
__TBB_store_relaxed(n->my_going, 2); 
if( n->my_state==STATE_UPGRADE_WAITING ) {
acquire_internal_lock();
queuing_rw_mutex::scoped_lock* tmp = tricky_pointer::fetch_and_store<tbb::release>(&(n->my_prev), NULL);
n->my_state = STATE_UPGRADE_LOSER;
__TBB_store_with_release(n->my_going,1);
unblock_or_wait_on_internal_lock(get_flag(tmp));
} else {
__TBB_ASSERT( my_state & (STATE_COMBINED_WAITINGREADER | STATE_WRITER), "unexpected state" );
__TBB_ASSERT( !( uintptr_t(__TBB_load_relaxed(n->my_prev)) & FLAG ), "use of corrupted pointer!" );
__TBB_store_relaxed(n->my_prev, (scoped_lock*)0);
__TBB_store_with_release(n->my_going,1);
}

} else { 

queuing_rw_mutex::scoped_lock *tmp = NULL;
retry:
queuing_rw_mutex::scoped_lock *pred = tricky_pointer::fetch_and_add<tbb::acquire>(&my_prev, FLAG);

if( pred ) {
if( !(pred->try_acquire_internal_lock()) )
{
tmp = tricky_pointer::compare_and_swap<tbb::release>(&my_prev, pred, tricky_pointer(pred) | FLAG );
if( !(uintptr_t(tmp) & FLAG) ) {
spin_wait_while_eq( my_prev, tricky_pointer(pred)|FLAG );
pred->release_internal_lock();
}

tmp = NULL;
goto retry;
}
__TBB_ASSERT(pred && pred->my_internal_lock==ACQUIRED, "predecessor's lock is not acquired");
__TBB_store_relaxed(my_prev, pred);
acquire_internal_lock();

__TBB_store_with_release(pred->my_next,static_cast<scoped_lock *>(NULL));

if( !__TBB_load_relaxed(my_next) && this != my_mutex->q_tail.compare_and_swap<tbb::release>(pred, this) ) {
spin_wait_while_eq( my_next, (void*)NULL );
}
__TBB_ASSERT( !get_flag(__TBB_load_relaxed(my_next)), "use of corrupted pointer" );

if( scoped_lock *const l_next = __TBB_load_with_acquire(my_next) ) { 
tmp = tricky_pointer::fetch_and_store<tbb::release>(&(l_next->my_prev), pred);
__TBB_ASSERT(__TBB_load_relaxed(my_prev)==pred, NULL);
__TBB_store_with_release(pred->my_next, my_next);
}
pred->release_internal_lock();

} else { 
acquire_internal_lock();  
scoped_lock* n = __TBB_load_with_acquire(my_next);
if( !n ) {
if( this != my_mutex->q_tail.compare_and_swap<tbb::release>(NULL, this) ) {
spin_wait_while_eq( my_next, (scoped_lock*)NULL );
n = __TBB_load_relaxed(my_next);
} else {
goto unlock_self;
}
}
__TBB_store_relaxed(n->my_going, 2); 
tmp = tricky_pointer::fetch_and_store<tbb::release>(&(n->my_prev), NULL);
__TBB_store_with_release(n->my_going,1);
}
unlock_self:
unblock_or_wait_on_internal_lock(get_flag(tmp));
}
done:
spin_wait_while_eq( my_going, 2 );

initialize();
}

bool queuing_rw_mutex::scoped_lock::downgrade_to_reader()
{
if ( my_state == STATE_ACTIVEREADER ) return true; 

ITT_NOTIFY(sync_releasing, my_mutex);
my_state = STATE_READER;
if( ! __TBB_load_relaxed(my_next) ) {
if( this==my_mutex->q_tail.load<full_fence>() ) {
unsigned short old_state = my_state.compare_and_swap<tbb::release>(STATE_ACTIVEREADER, STATE_READER);
if( old_state==STATE_READER )
return true; 
}

spin_wait_while_eq( my_next, (void*)NULL );
}
scoped_lock *const n = __TBB_load_with_acquire(my_next);
__TBB_ASSERT( n, "still no successor at this point!" );
if( n->my_state & STATE_COMBINED_WAITINGREADER )
__TBB_store_with_release(n->my_going,1);
else if( n->my_state==STATE_UPGRADE_WAITING )
n->my_state = STATE_UPGRADE_LOSER;
my_state = STATE_ACTIVEREADER;
return true;
}

bool queuing_rw_mutex::scoped_lock::upgrade_to_writer()
{
if ( my_state == STATE_WRITER ) return true; 

queuing_rw_mutex::scoped_lock * tmp;
queuing_rw_mutex::scoped_lock * me = this;

ITT_NOTIFY(sync_releasing, my_mutex);
my_state = STATE_UPGRADE_REQUESTED;
requested:
__TBB_ASSERT( !(uintptr_t(__TBB_load_relaxed(my_next)) & FLAG), "use of corrupted pointer!" );
acquire_internal_lock();
if( this != my_mutex->q_tail.compare_and_swap<tbb::release>(tricky_pointer(me)|FLAG, this) ) {
spin_wait_while_eq( my_next, (void*)NULL );
queuing_rw_mutex::scoped_lock * n;
n = tricky_pointer::fetch_and_add<tbb::acquire>(&my_next, FLAG);
unsigned short n_state = n->my_state;

if( n_state & STATE_COMBINED_WAITINGREADER )
__TBB_store_with_release(n->my_going,1);
tmp = tricky_pointer::fetch_and_store<tbb::release>(&(n->my_prev), this);
unblock_or_wait_on_internal_lock(get_flag(tmp));
if( n_state & (STATE_COMBINED_READER | STATE_UPGRADE_REQUESTED) ) {
tmp = tricky_pointer(n)|FLAG;
for( atomic_backoff b; __TBB_load_relaxed(my_next)==tmp; b.pause() ) {
if( my_state & STATE_COMBINED_UPGRADING ) {
if( __TBB_load_with_acquire(my_next)==tmp )
__TBB_store_relaxed(my_next, n);
goto waiting;
}
}
__TBB_ASSERT(__TBB_load_relaxed(my_next) != (tricky_pointer(n)|FLAG), NULL);
goto requested;
} else {
__TBB_ASSERT( n_state & (STATE_WRITER | STATE_UPGRADE_WAITING), "unexpected state");
__TBB_ASSERT( (tricky_pointer(n)|FLAG) == __TBB_load_relaxed(my_next), NULL);
__TBB_store_relaxed(my_next, n);
}
} else {

release_internal_lock();
} 
my_state.compare_and_swap<tbb::acquire>(STATE_UPGRADE_WAITING, STATE_UPGRADE_REQUESTED);

waiting:
__TBB_ASSERT( !( intptr_t(__TBB_load_relaxed(my_next)) & FLAG ), "use of corrupted pointer!" );
__TBB_ASSERT( my_state & STATE_COMBINED_UPGRADING, "wrong state at upgrade waiting_retry" );
__TBB_ASSERT( me==this, NULL );
ITT_NOTIFY(sync_prepare, my_mutex);

my_mutex->q_tail.compare_and_swap<tbb::release>( this, tricky_pointer(me)|FLAG );
queuing_rw_mutex::scoped_lock * pred;
pred = tricky_pointer::fetch_and_add<tbb::acquire>(&my_prev, FLAG);
if( pred ) {
bool success = pred->try_acquire_internal_lock();
pred->my_state.compare_and_swap<tbb::release>(STATE_UPGRADE_WAITING, STATE_UPGRADE_REQUESTED);
if( !success ) {
tmp = tricky_pointer::compare_and_swap<tbb::release>(&my_prev, pred, tricky_pointer(pred)|FLAG );
if( uintptr_t(tmp) & FLAG ) {
spin_wait_while_eq(my_prev, pred);
pred = __TBB_load_relaxed(my_prev);
} else {
spin_wait_while_eq( my_prev, tricky_pointer(pred)|FLAG );
pred->release_internal_lock();
}
} else {
__TBB_store_relaxed(my_prev, pred);
pred->release_internal_lock();
spin_wait_while_eq(my_prev, pred);
pred = __TBB_load_relaxed(my_prev);
}
if( pred )
goto waiting;
} else {
__TBB_store_relaxed(my_prev, pred);
}
__TBB_ASSERT( !pred && !__TBB_load_relaxed(my_prev), NULL );

wait_for_release_of_internal_lock();
spin_wait_while_eq( my_going, 2 );

__TBB_control_consistency_helper(); 

bool result = ( my_state != STATE_UPGRADE_LOSER );
my_state = STATE_WRITER;
__TBB_store_relaxed(my_going, 1);

ITT_NOTIFY(sync_acquired, my_mutex);
return result;
}

void queuing_rw_mutex::internal_construct() {
ITT_SYNC_CREATE(this, _T("tbb::queuing_rw_mutex"), _T(""));
}

} 
