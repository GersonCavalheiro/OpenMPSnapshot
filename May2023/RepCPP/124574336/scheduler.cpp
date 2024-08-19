

#include "custom_scheduler.h"
#include "scheduler_utility.h"
#include "governor.h"
#include "market.h"
#include "arena.h"
#include "mailbox.h"
#include "observer_proxy.h"
#include "tbb/tbb_machine.h"
#include "tbb/atomic.h"

namespace tbb {
namespace internal {



extern generic_scheduler* (*AllocateSchedulerPtr)( market&, bool );

inline generic_scheduler* allocate_scheduler ( market& m, bool genuine ) {
return AllocateSchedulerPtr( m, genuine);
}

#if __TBB_TASK_GROUP_CONTEXT
context_state_propagation_mutex_type the_context_state_propagation_mutex;

uintptr_t the_context_state_propagation_epoch = 0;


static task_group_context the_dummy_context(task_group_context::isolated);
#endif 

void Scheduler_OneTimeInitialization ( bool itt_present ) {
AllocateSchedulerPtr = itt_present ? &custom_scheduler<DefaultSchedulerTraits>::allocate_scheduler :
&custom_scheduler<IntelSchedulerTraits>::allocate_scheduler;
#if __TBB_TASK_GROUP_CONTEXT
__TBB_ASSERT(!(task_group_context::low_unused_state_bit & (task_group_context::low_unused_state_bit-1)), NULL);
the_dummy_context.my_state = task_group_context::low_unused_state_bit;
#if __TBB_TASK_PRIORITY
the_dummy_context.my_priority = num_priority_levels - 1;
#endif 
#endif 
}


scheduler::~scheduler( ) {}


#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable:4355)
#endif

generic_scheduler::generic_scheduler( market& m, bool genuine )
: my_market(&m)
, my_random(this)
, my_ref_count(1)
#if __TBB_PREVIEW_RESUMABLE_TASKS
, my_co_context(m.worker_stack_size(), genuine ? NULL : this)
#endif
, my_small_task_count(1)   
#if __TBB_SURVIVE_THREAD_SWITCH && TBB_USE_ASSERT
, my_cilk_state(cs_none)
#endif 
{
__TBB_ASSERT( !my_arena_index, "constructor expects the memory being zero-initialized" );
__TBB_ASSERT( governor::is_set(NULL), "scheduler is already initialized for this thread" );

my_innermost_running_task = my_dummy_task = &allocate_task( sizeof(task), __TBB_CONTEXT_ARG(NULL, &the_dummy_context) );
#if __TBB_PREVIEW_CRITICAL_TASKS
my_properties.has_taken_critical_task = false;
#endif
#if __TBB_PREVIEW_RESUMABLE_TASKS
my_properties.genuine = genuine;
my_current_is_recalled = NULL;
my_post_resume_action = PRA_NONE;
my_post_resume_arg = NULL;
my_wait_task = NULL;
#else
suppress_unused_warning(genuine);
#endif
my_properties.outermost = true;
#if __TBB_TASK_PRIORITY
my_ref_top_priority = &m.my_global_top_priority;
my_ref_reload_epoch = &m.my_global_reload_epoch;
#endif 
#if __TBB_TASK_GROUP_CONTEXT
my_context_state_propagation_epoch = the_context_state_propagation_epoch;
my_context_list_head.my_prev = &my_context_list_head;
my_context_list_head.my_next = &my_context_list_head;
ITT_SYNC_CREATE(&my_context_list_mutex, SyncType_Scheduler, SyncObj_ContextsList);
#endif 
ITT_SYNC_CREATE(&my_dummy_task->prefix().ref_count, SyncType_Scheduler, SyncObj_WorkerLifeCycleMgmt);
ITT_SYNC_CREATE(&my_return_list, SyncType_Scheduler, SyncObj_TaskReturnList);
}

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning(pop)
#endif 

#if TBB_USE_ASSERT > 1
void generic_scheduler::assert_task_pool_valid() const {
if ( !my_arena_slot )
return;
acquire_task_pool();
task** tp = my_arena_slot->task_pool_ptr;
if ( my_arena_slot->my_task_pool_size )
__TBB_ASSERT( my_arena_slot->my_task_pool_size >= min_task_pool_size, NULL );
const size_t H = __TBB_load_relaxed(my_arena_slot->head); 
const size_t T = __TBB_load_relaxed(my_arena_slot->tail); 
__TBB_ASSERT( H <= T, NULL );
for ( size_t i = 0; i < H; ++i )
__TBB_ASSERT( tp[i] == poisoned_ptr, "Task pool corrupted" );
for ( size_t i = H; i < T; ++i ) {
if ( tp[i] ) {
assert_task_valid( tp[i] );
__TBB_ASSERT( tp[i]->prefix().state == task::ready ||
tp[i]->prefix().extra_state == es_task_proxy, "task in the deque has invalid state" );
}
}
for ( size_t i = T; i < my_arena_slot->my_task_pool_size; ++i )
__TBB_ASSERT( tp[i] == poisoned_ptr, "Task pool corrupted" );
release_task_pool();
}
#endif 

void generic_scheduler::init_stack_info () {
__TBB_ASSERT( !my_stealing_threshold, "Stealing threshold has already been calculated" );
size_t  stack_size = my_market->worker_stack_size();
#if USE_WINTHREAD
#if defined(_MSC_VER)&&_MSC_VER<1400 && !_WIN64
NT_TIB  *pteb;
__asm mov eax, fs:[0x18]
__asm mov pteb, eax
#else
NT_TIB  *pteb = (NT_TIB*)NtCurrentTeb();
#endif
__TBB_ASSERT( &pteb < pteb->StackBase && &pteb > pteb->StackLimit, "invalid stack info in TEB" );
__TBB_ASSERT( stack_size >0, "stack_size not initialized?" );
if ( is_worker() || stack_size < MByte )
my_stealing_threshold = (uintptr_t)pteb->StackBase - stack_size / 2;
else
my_stealing_threshold = (uintptr_t)pteb->StackBase - MByte / 2;
#else 
void    *stack_base = &stack_size;
#if __linux__ && !__bg__
#if __TBB_ipf
void    *rsb_base = __TBB_get_bsp();
#endif
size_t  np_stack_size = 0;
void    *stack_limit = NULL;

#if __TBB_PREVIEW_RESUMABLE_TASKS
if ( !my_properties.genuine ) {
stack_limit = my_co_context.get_stack_limit();
__TBB_ASSERT( (uintptr_t)stack_base > (uintptr_t)stack_limit, "stack size must be positive" );
stack_size = size_t((char*)stack_base - (char*)stack_limit);
}
#endif

pthread_attr_t  np_attr_stack;
if( !stack_limit && 0 == pthread_getattr_np(pthread_self(), &np_attr_stack) ) {
if ( 0 == pthread_attr_getstack(&np_attr_stack, &stack_limit, &np_stack_size) ) {
#if __TBB_ipf
pthread_attr_t  attr_stack;
if ( 0 == pthread_attr_init(&attr_stack) ) {
if ( 0 == pthread_attr_getstacksize(&attr_stack, &stack_size) ) {
if ( np_stack_size < stack_size ) {
rsb_base = stack_limit;
stack_size = np_stack_size/2;
stack_limit = (char*)stack_limit + stack_size;
}
}
pthread_attr_destroy(&attr_stack);
}
my_rsb_stealing_threshold = (uintptr_t)((char*)rsb_base + stack_size/2);
#endif 
stack_size = size_t((char*)stack_base - (char*)stack_limit);
}
pthread_attr_destroy(&np_attr_stack);
}
#endif 
__TBB_ASSERT( stack_size>0, "stack size must be positive" );
my_stealing_threshold = (uintptr_t)((char*)stack_base - stack_size/2);
#endif 
}

#if __TBB_TASK_GROUP_CONTEXT

void generic_scheduler::cleanup_local_context_list () {
bool wait_for_concurrent_destroyers_to_leave = false;
uintptr_t local_count_snapshot = my_context_state_propagation_epoch;
my_local_ctx_list_update.store<relaxed>(1);
{
spin_mutex::scoped_lock lock;
atomic_fence();
if ( my_nonlocal_ctx_list_update.load<relaxed>() || local_count_snapshot != the_context_state_propagation_epoch )
lock.acquire(my_context_list_mutex);
context_list_node_t *node = my_context_list_head.my_next;
while ( node != &my_context_list_head ) {
task_group_context &ctx = __TBB_get_object_ref(task_group_context, my_node, node);
__TBB_ASSERT( __TBB_load_relaxed(ctx.my_kind) != task_group_context::binding_required, "Only a context bound to a root task can be detached" );
node = node->my_next;
__TBB_ASSERT( is_alive(ctx.my_version_and_traits), "Walked into a destroyed context while detaching contexts from the local list" );
if ( internal::as_atomic(ctx.my_kind).fetch_and_store(task_group_context::detached) == task_group_context::dying )
wait_for_concurrent_destroyers_to_leave = true;
}
}
my_local_ctx_list_update.store<release>(0);
if ( wait_for_concurrent_destroyers_to_leave )
spin_wait_until_eq( my_nonlocal_ctx_list_update, 0u );
}
#endif 

void generic_scheduler::destroy() {
__TBB_ASSERT(my_small_task_count == 0, "The scheduler is still in use.");
this->~generic_scheduler();
#if TBB_USE_DEBUG
memset((void*)this, -1, sizeof(generic_scheduler));
#endif
NFS_Free(this);
}

void generic_scheduler::cleanup_scheduler() {
__TBB_ASSERT( !my_arena_slot, NULL );
#if __TBB_TASK_PRIORITY
__TBB_ASSERT( my_offloaded_tasks == NULL, NULL );
#endif
#if __TBB_PREVIEW_CRITICAL_TASKS
__TBB_ASSERT( !my_properties.has_taken_critical_task, "Critical tasks miscount." );
#endif
#if __TBB_TASK_GROUP_CONTEXT
cleanup_local_context_list();
#endif 
free_task<small_local_task>( *my_dummy_task );

#if __TBB_HOARD_NONLOCAL_TASKS
while( task* t = my_nonlocal_free_list ) {
task_prefix& p = t->prefix();
my_nonlocal_free_list = p.next;
__TBB_ASSERT( p.origin && p.origin!=this, NULL );
free_nonlocal_small_task(*t);
}
#endif
intptr_t k = 1;
for(;;) {
while( task* t = my_free_list ) {
my_free_list = t->prefix().next;
deallocate_task(*t);
++k;
}
if( my_return_list==plugged_return_list() )
break;
my_free_list = (task*)__TBB_FetchAndStoreW( &my_return_list, (intptr_t)plugged_return_list() );
}
#if __TBB_COUNT_TASK_NODES
my_market->update_task_node_count( my_task_node_count );
#endif 
__TBB_ASSERT( my_small_task_count>=k, "my_small_task_count corrupted" );
governor::sign_off(this);
if( __TBB_FetchAndAddW( &my_small_task_count, -k )==k )
destroy();
}

task& generic_scheduler::allocate_task( size_t number_of_bytes,
__TBB_CONTEXT_ARG(task* parent, task_group_context* context) ) {
GATHER_STATISTIC(++my_counters.active_tasks);
task *t;
if( number_of_bytes<=quick_task_size ) {
#if __TBB_HOARD_NONLOCAL_TASKS
if( (t = my_nonlocal_free_list) ) {
GATHER_STATISTIC(--my_counters.free_list_length);
__TBB_ASSERT( t->state()==task::freed, "free list of tasks is corrupted" );
my_nonlocal_free_list = t->prefix().next;
} else
#endif
if( (t = my_free_list) ) {
GATHER_STATISTIC(--my_counters.free_list_length);
__TBB_ASSERT( t->state()==task::freed, "free list of tasks is corrupted" );
my_free_list = t->prefix().next;
} else if( my_return_list ) {
t = (task*)__TBB_FetchAndStoreW( &my_return_list, 0 ); 
__TBB_ASSERT( t, "another thread emptied the my_return_list" );
__TBB_ASSERT( t->prefix().origin==this, "task returned to wrong my_return_list" );
ITT_NOTIFY( sync_acquired, &my_return_list );
my_free_list = t->prefix().next;
} else {
t = (task*)((char*)NFS_Allocate( 1, task_prefix_reservation_size+quick_task_size, NULL ) + task_prefix_reservation_size );
#if __TBB_COUNT_TASK_NODES
++my_task_node_count;
#endif 
t->prefix().origin = this;
t->prefix().next = 0;
++my_small_task_count;
}
#if __TBB_PREFETCHING
task *t_next = t->prefix().next;
if( !t_next ) { 
#if __TBB_HOARD_NONLOCAL_TASKS
if( my_free_list )
t_next = my_free_list;
else
#endif
if( my_return_list ) 
t_next = my_free_list = (task*)__TBB_FetchAndStoreW( &my_return_list, 0 );
}
if( t_next ) { 
__TBB_cl_prefetch(t_next);
__TBB_cl_prefetch(&t_next->prefix());
}
#endif 
} else {
GATHER_STATISTIC(++my_counters.big_tasks);
t = (task*)((char*)NFS_Allocate( 1, task_prefix_reservation_size+number_of_bytes, NULL ) + task_prefix_reservation_size );
#if __TBB_COUNT_TASK_NODES
++my_task_node_count;
#endif 
t->prefix().origin = NULL;
}
task_prefix& p = t->prefix();
#if __TBB_TASK_GROUP_CONTEXT
p.context = context;
#endif 
p.owner = this;
p.ref_count = 0;
p.depth = 0;
p.parent = parent;
p.extra_state = 0;
p.affinity = 0;
p.state = task::allocated;
__TBB_ISOLATION_EXPR( p.isolation = no_isolation );
return *t;
}

void generic_scheduler::free_nonlocal_small_task( task& t ) {
__TBB_ASSERT( t.state()==task::freed, NULL );
generic_scheduler& s = *static_cast<generic_scheduler*>(t.prefix().origin);
__TBB_ASSERT( &s!=this, NULL );
for(;;) {
task* old = s.my_return_list;
if( old==plugged_return_list() )
break;
t.prefix().next = old;
ITT_NOTIFY( sync_releasing, &s.my_return_list );
if( as_atomic(s.my_return_list).compare_and_swap(&t, old )==old ) {
#if __TBB_PREFETCHING
__TBB_cl_evict(&t.prefix());
__TBB_cl_evict(&t);
#endif
return;
}
}
deallocate_task(t);
if( __TBB_FetchAndDecrementWrelease( &s.my_small_task_count )==1 ) {
s.destroy();
}
}

inline size_t generic_scheduler::prepare_task_pool ( size_t num_tasks ) {
size_t T = __TBB_load_relaxed(my_arena_slot->tail); 
if ( T + num_tasks <= my_arena_slot->my_task_pool_size )
return T;

size_t new_size = num_tasks;

if ( !my_arena_slot->my_task_pool_size ) {
__TBB_ASSERT( !is_task_pool_published() && is_quiescent_local_task_pool_reset(), NULL );
__TBB_ASSERT( !my_arena_slot->task_pool_ptr, NULL );
if ( num_tasks < min_task_pool_size ) new_size = min_task_pool_size;
my_arena_slot->allocate_task_pool( new_size );
return 0;
}

acquire_task_pool();
size_t H = __TBB_load_relaxed( my_arena_slot->head ); 
task** task_pool = my_arena_slot->task_pool_ptr;;
__TBB_ASSERT( my_arena_slot->my_task_pool_size >= min_task_pool_size, NULL );
for ( size_t i = H; i < T; ++i )
if ( task_pool[i] ) ++new_size;
bool allocate = new_size > my_arena_slot->my_task_pool_size - min_task_pool_size/4;
if ( allocate ) {
if ( new_size < 2 * my_arena_slot->my_task_pool_size )
new_size = 2 * my_arena_slot->my_task_pool_size;
my_arena_slot->allocate_task_pool( new_size ); 
}
size_t T1 = 0;
for ( size_t i = H; i < T; ++i )
if ( task_pool[i] )
my_arena_slot->task_pool_ptr[T1++] = task_pool[i];
if ( allocate )
NFS_Free( task_pool );
else
my_arena_slot->fill_with_canary_pattern( T1, my_arena_slot->tail );
commit_relocated_tasks( T1 );
assert_task_pool_valid();
return T1;
}


inline void generic_scheduler::acquire_task_pool() const {
if ( !is_task_pool_published() )
return; 
bool sync_prepare_done = false;
for( atomic_backoff b;;b.pause() ) {
#if TBB_USE_ASSERT
__TBB_ASSERT( my_arena_slot == my_arena->my_slots + my_arena_index, "invalid arena slot index" );
task** tp = my_arena_slot->task_pool;
__TBB_ASSERT( tp == LockedTaskPool || tp == my_arena_slot->task_pool_ptr, "slot ownership corrupt?" );
#endif
if( my_arena_slot->task_pool != LockedTaskPool &&
as_atomic(my_arena_slot->task_pool).compare_and_swap(LockedTaskPool, my_arena_slot->task_pool_ptr ) == my_arena_slot->task_pool_ptr )
{
ITT_NOTIFY(sync_acquired, my_arena_slot);
break;
}
else if( !sync_prepare_done ) {
ITT_NOTIFY(sync_prepare, my_arena_slot);
sync_prepare_done = true;
}
}
__TBB_ASSERT( my_arena_slot->task_pool == LockedTaskPool, "not really acquired task pool" );
} 

inline void generic_scheduler::release_task_pool() const {
if ( !is_task_pool_published() )
return; 
__TBB_ASSERT( my_arena_slot, "we are not in arena" );
__TBB_ASSERT( my_arena_slot->task_pool == LockedTaskPool, "arena slot is not locked" );
ITT_NOTIFY(sync_releasing, my_arena_slot);
__TBB_store_with_release( my_arena_slot->task_pool, my_arena_slot->task_pool_ptr );
}


inline task** generic_scheduler::lock_task_pool( arena_slot* victim_arena_slot ) const {
task** victim_task_pool;
bool sync_prepare_done = false;
for( atomic_backoff backoff;; ) {
victim_task_pool = victim_arena_slot->task_pool;
if ( victim_task_pool == EmptyTaskPool ) {
if( sync_prepare_done )
ITT_NOTIFY(sync_cancel, victim_arena_slot);
break;
}
if( victim_task_pool != LockedTaskPool &&
as_atomic(victim_arena_slot->task_pool).compare_and_swap(LockedTaskPool, victim_task_pool ) == victim_task_pool )
{
ITT_NOTIFY(sync_acquired, victim_arena_slot);
break;
}
else if( !sync_prepare_done ) {
ITT_NOTIFY(sync_prepare, victim_arena_slot);
sync_prepare_done = true;
}
GATHER_STATISTIC( ++my_counters.thieves_conflicts );
#if __TBB_STEALING_ABORT_ON_CONTENTION
if(!backoff.bounded_pause()) {
if(my_arena->my_limit >= 16)
return EmptyTaskPool;
__TBB_Yield();
}
#else
backoff.pause();
#endif
}
__TBB_ASSERT( victim_task_pool == EmptyTaskPool ||
(victim_arena_slot->task_pool == LockedTaskPool && victim_task_pool != LockedTaskPool),
"not really locked victim's task pool?" );
return victim_task_pool;
} 

inline void generic_scheduler::unlock_task_pool( arena_slot* victim_arena_slot,
task** victim_task_pool ) const {
__TBB_ASSERT( victim_arena_slot, "empty victim arena slot pointer" );
__TBB_ASSERT( victim_arena_slot->task_pool == LockedTaskPool, "victim arena slot is not locked" );
ITT_NOTIFY(sync_releasing, victim_arena_slot);
__TBB_store_with_release( victim_arena_slot->task_pool, victim_task_pool );
}


inline task* generic_scheduler::prepare_for_spawning( task* t ) {
__TBB_ASSERT( t->state()==task::allocated, "attempt to spawn task that is not in 'allocated' state" );
t->prefix().state = task::ready;
#if TBB_USE_ASSERT
if( task* parent = t->parent() ) {
internal::reference_count ref_count = parent->prefix().ref_count;
__TBB_ASSERT( ref_count>=0, "attempt to spawn task whose parent has a ref_count<0" );
__TBB_ASSERT( ref_count!=0, "attempt to spawn task whose parent has a ref_count==0 (forgot to set_ref_count?)" );
parent->prefix().extra_state |= es_ref_count_active;
}
#endif 
affinity_id dst_thread = t->prefix().affinity;
__TBB_ASSERT( dst_thread == 0 || is_version_3_task(*t),
"backwards compatibility to TBB 2.0 tasks is broken" );
#if __TBB_TASK_ISOLATION
isolation_tag isolation = my_innermost_running_task->prefix().isolation;
t->prefix().isolation = isolation;
#endif 
if( dst_thread != 0 && dst_thread != my_affinity_id ) {
task_proxy& proxy = (task_proxy&)allocate_task( sizeof(task_proxy),
__TBB_CONTEXT_ARG(NULL, NULL) );
proxy.prefix().extra_state = es_task_proxy;
proxy.outbox = &my_arena->mailbox(dst_thread);
proxy.task_and_tag = intptr_t(t) | task_proxy::location_mask;
#if __TBB_TASK_PRIORITY
poison_pointer( proxy.prefix().context );
#endif 
__TBB_ISOLATION_EXPR( proxy.prefix().isolation = isolation );
ITT_NOTIFY( sync_releasing, proxy.outbox );
if ( proxy.outbox->push(&proxy) )
return &proxy;
free_task<small_task>(proxy);
}
return t;
}

#if __TBB_PREVIEW_CRITICAL_TASKS
bool generic_scheduler::handled_as_critical( task& t ) {
if( !internal::is_critical( t ) )
return false;
#if __TBB_TASK_ISOLATION
t.prefix().isolation = my_innermost_running_task->prefix().isolation;
#endif
ITT_NOTIFY(sync_releasing, &my_arena->my_critical_task_stream);
__TBB_ASSERT( my_arena, "Must be attached to the arena." );
__TBB_ASSERT( my_arena_slot, "Must occupy a slot in the attached arena" );
my_arena->my_critical_task_stream.push(
&t, 0, tbb::internal::subsequent_lane_selector(my_arena_slot->hint_for_critical) );
return true;
}
#endif 


void generic_scheduler::local_spawn( task* first, task*& next ) {
__TBB_ASSERT( first, NULL );
__TBB_ASSERT( governor::is_set(this), NULL );
#if __TBB_TODO
#endif
if ( &first->prefix().next == &next ) {
#if __TBB_TODO
#endif
#if __TBB_PREVIEW_CRITICAL_TASKS
if( !handled_as_critical( *first ) )
#endif
{
size_t T = prepare_task_pool( 1 );
my_arena_slot->task_pool_ptr[T] = prepare_for_spawning( first );
commit_spawned_tasks( T + 1 );
if ( !is_task_pool_published() )
publish_task_pool();
}
}
else {
#if __TBB_TODO
#endif
task *arr[min_task_pool_size];
fast_reverse_vector<task*> tasks(arr, min_task_pool_size);
task *t_next = NULL;
for( task* t = first; ; t = t_next ) {
bool end = &t->prefix().next == &next;
t_next = t->prefix().next;
#if __TBB_PREVIEW_CRITICAL_TASKS
if( !handled_as_critical( *t ) )
#endif
tasks.push_back( prepare_for_spawning(t) );
if( end )
break;
}
if( size_t num_tasks = tasks.size() ) {
size_t T = prepare_task_pool( num_tasks );
tasks.copy_memory( my_arena_slot->task_pool_ptr + T );
commit_spawned_tasks( T + num_tasks );
if ( !is_task_pool_published() )
publish_task_pool();
}
}
my_arena->advertise_new_work<arena::work_spawned>();
assert_task_pool_valid();
}

void generic_scheduler::local_spawn_root_and_wait( task* first, task*& next ) {
__TBB_ASSERT( governor::is_set(this), NULL );
__TBB_ASSERT( first, NULL );
auto_empty_task dummy( __TBB_CONTEXT_ARG(this, first->prefix().context) );
internal::reference_count n = 0;
for( task* t=first; ; t=t->prefix().next ) {
++n;
__TBB_ASSERT( !t->prefix().parent, "not a root task, or already running" );
t->prefix().parent = &dummy;
if( &t->prefix().next==&next ) break;
#if __TBB_TASK_GROUP_CONTEXT
__TBB_ASSERT( t->prefix().context == t->prefix().next->prefix().context,
"all the root tasks in list must share the same context");
#endif 
}
dummy.prefix().ref_count = n+1;
if( n>1 )
local_spawn( first->prefix().next, next );
local_wait_for_all( dummy, first );
}

void tbb::internal::generic_scheduler::spawn( task& first, task*& next ) {
governor::local_scheduler()->local_spawn( &first, next );
}

void tbb::internal::generic_scheduler::spawn_root_and_wait( task& first, task*& next ) {
governor::local_scheduler()->local_spawn_root_and_wait( &first, next );
}

void tbb::internal::generic_scheduler::enqueue( task& t, void* prio ) {
generic_scheduler *s = governor::local_scheduler();
__TBB_ASSERT( s->my_arena, "thread is not in any arena" );
s->my_arena->enqueue_task(t, (intptr_t)prio, s->my_random );
}

#if __TBB_TASK_PRIORITY
class auto_indicator : no_copy {
volatile bool& my_indicator;
public:
auto_indicator ( volatile bool& indicator ) : my_indicator(indicator) { my_indicator = true ;}
~auto_indicator () { my_indicator = false; }
};

task *generic_scheduler::get_task_and_activate_task_pool( size_t H0, __TBB_ISOLATION_ARG( size_t T0, isolation_tag isolation ) ) {
__TBB_ASSERT( is_local_task_pool_quiescent(), NULL );

task *t = NULL;
#if __TBB_TASK_ISOLATION
size_t T = T0;
bool tasks_omitted = false;
while ( !t && T>H0 ) {
t = get_task( --T, isolation, tasks_omitted );
if ( !tasks_omitted ) {
poison_pointer( my_arena_slot->task_pool_ptr[T] );
--T0;
}
}
if ( t && tasks_omitted ) {
my_arena_slot->task_pool_ptr[T] = NULL;
if ( T == H0 ) {
++H0;
poison_pointer( my_arena_slot->task_pool_ptr[T] );
}
}
#else
while ( !t && T0 ) {
t = get_task( --T0 );
poison_pointer( my_arena_slot->task_pool_ptr[T0] );
}
#endif 

if ( H0 < T0 ) {
__TBB_store_relaxed( my_arena_slot->head, H0 );
__TBB_store_relaxed( my_arena_slot->tail, T0 );
if ( is_task_pool_published() )
release_task_pool();
else
publish_task_pool();
} else {
__TBB_store_relaxed( my_arena_slot->head, 0 );
__TBB_store_relaxed( my_arena_slot->tail, 0 );
if ( is_task_pool_published() )
leave_task_pool();
}

#if __TBB_TASK_ISOLATION
if ( tasks_omitted && my_innermost_running_task == t ) {
assert_task_valid( t );
t->note_affinity( my_affinity_id );
}
#endif 

assert_task_pool_valid();
return t;
}

task* generic_scheduler::winnow_task_pool( __TBB_ISOLATION_EXPR( isolation_tag isolation ) ) {
GATHER_STATISTIC( ++my_counters.prio_winnowings );
__TBB_ASSERT( is_task_pool_published(), NULL );
__TBB_ASSERT( my_offloaded_tasks, "At least one task is expected to be already offloaded" );
auto_indicator indicator( my_pool_reshuffling_pending );

acquire_task_pool();
size_t T0 = __TBB_load_relaxed( my_arena_slot->tail );
size_t H0 = __TBB_load_relaxed( my_arena_slot->head );
size_t T1 = 0;
for ( size_t src = H0; src<T0; ++src ) {
if ( task *t = my_arena_slot->task_pool_ptr[src] ) {
if ( !is_proxy( *t ) ) {
intptr_t p = priority( *t );
if ( p<*my_ref_top_priority ) {
offload_task( *t, p );
continue;
}
}
my_arena_slot->task_pool_ptr[T1++] = t;
}
}
__TBB_ASSERT( T1<=T0, NULL );

my_arena_slot->fill_with_canary_pattern( max( T1, H0 ), T0 );
return get_task_and_activate_task_pool( 0, __TBB_ISOLATION_ARG( T1, isolation ) );
}

task* generic_scheduler::reload_tasks ( task*& offloaded_tasks, task**& offloaded_task_list_link, __TBB_ISOLATION_ARG( intptr_t top_priority, isolation_tag isolation ) ) {
GATHER_STATISTIC( ++my_counters.prio_reloads );
#if __TBB_TASK_ISOLATION
acquire_task_pool();
#else
__TBB_ASSERT( !is_task_pool_published(), NULL );
#endif
task *arr[min_task_pool_size];
fast_reverse_vector<task*> tasks(arr, min_task_pool_size);
task **link = &offloaded_tasks;
while ( task *t = *link ) {
task** next_ptr = &t->prefix().next_offloaded;
__TBB_ASSERT( !is_proxy(*t), "The proxy tasks cannot be offloaded" );
if ( priority(*t) >= top_priority ) {
tasks.push_back( t );
task* next = *next_ptr;
t->prefix().owner = this;
__TBB_ASSERT( t->prefix().state == task::ready, NULL );
*link = next;
}
else {
link = next_ptr;
}
}
if ( link == &offloaded_tasks ) {
offloaded_tasks = NULL;
#if TBB_USE_ASSERT
offloaded_task_list_link = NULL;
#endif 
}
else {
__TBB_ASSERT( link, NULL );
*link = NULL;
offloaded_task_list_link = link;
}
__TBB_ASSERT( link, NULL );
size_t num_tasks = tasks.size();
if ( !num_tasks ) {
__TBB_ISOLATION_EXPR( release_task_pool() );
return NULL;
}

GATHER_STATISTIC( ++my_counters.prio_tasks_reloaded );
size_t T = prepare_task_pool( num_tasks );
tasks.copy_memory( my_arena_slot->task_pool_ptr + T );

task *t = get_task_and_activate_task_pool( __TBB_load_relaxed( my_arena_slot->head ), __TBB_ISOLATION_ARG( T + num_tasks, isolation ) );
if ( t ) --num_tasks;
if ( num_tasks )
my_arena->advertise_new_work<arena::work_spawned>();

return t;
}

task* generic_scheduler::reload_tasks( __TBB_ISOLATION_EXPR( isolation_tag isolation ) ) {
uintptr_t reload_epoch = *my_ref_reload_epoch;
__TBB_ASSERT( my_offloaded_tasks, NULL );
__TBB_ASSERT( my_local_reload_epoch <= reload_epoch
|| my_local_reload_epoch - reload_epoch > uintptr_t(-1)/2,
"Reload epoch counter overflow?" );
if ( my_local_reload_epoch == reload_epoch )
return NULL;
__TBB_ASSERT( my_offloaded_tasks, NULL );
intptr_t top_priority = effective_reference_priority();
__TBB_ASSERT( (uintptr_t)top_priority < (uintptr_t)num_priority_levels, NULL );
task *t = reload_tasks( my_offloaded_tasks, my_offloaded_task_list_tail_link, __TBB_ISOLATION_ARG( top_priority, isolation ) );
if ( my_offloaded_tasks && (my_arena->my_bottom_priority >= top_priority || !my_arena->my_num_workers_requested) ) {

my_market->update_arena_priority( *my_arena, priority(*my_offloaded_tasks) );
my_arena->advertise_new_work<arena::wakeup>();
}
my_local_reload_epoch = reload_epoch;
return t;
}
#endif 

#if __TBB_TASK_ISOLATION
inline task* generic_scheduler::get_task( size_t T, isolation_tag isolation, bool& tasks_omitted )
#else
inline task* generic_scheduler::get_task( size_t T )
#endif 
{
__TBB_ASSERT( __TBB_load_relaxed( my_arena_slot->tail ) <= T
|| is_local_task_pool_quiescent(), "Is it safe to get a task at position T?" );

task* result = my_arena_slot->task_pool_ptr[T];
__TBB_ASSERT( !is_poisoned( result ), "The poisoned task is going to be processed" );
#if __TBB_TASK_ISOLATION
if ( !result )
return NULL;

bool omit = isolation != no_isolation && isolation != result->prefix().isolation;
if ( !omit && !is_proxy( *result ) )
return result;
else if ( omit ) {
tasks_omitted = true;
return NULL;
}
#else
poison_pointer( my_arena_slot->task_pool_ptr[T] );
if ( !result || !is_proxy( *result ) )
return result;
#endif 

task_proxy& tp = static_cast<task_proxy&>(*result);
if ( task *t = tp.extract_task<task_proxy::pool_bit>() ) {
GATHER_STATISTIC( ++my_counters.proxies_executed );
__TBB_ASSERT( is_version_3_task( *t ), "backwards compatibility with TBB 2.0 broken" );
my_innermost_running_task = t; 
#if __TBB_TASK_ISOLATION
if ( !tasks_omitted )
#endif 
{
poison_pointer( my_arena_slot->task_pool_ptr[T] );
t->note_affinity( my_affinity_id );
}
return t;
}

free_task<small_task>( tp );
#if __TBB_TASK_ISOLATION
if ( tasks_omitted )
my_arena_slot->task_pool_ptr[T] = NULL;
#endif 
return NULL;
}

inline task* generic_scheduler::get_task( __TBB_ISOLATION_EXPR( isolation_tag isolation ) ) {
__TBB_ASSERT( is_task_pool_published(), NULL );
size_t T0 = __TBB_load_relaxed( my_arena_slot->tail );
size_t H0 = (size_t)-1, T = T0;
task* result = NULL;
bool task_pool_empty = false;
__TBB_ISOLATION_EXPR( bool tasks_omitted = false );
do {
__TBB_ASSERT( !result, NULL );
__TBB_store_relaxed( my_arena_slot->tail, --T );
atomic_fence();
if ( (intptr_t)__TBB_load_relaxed( my_arena_slot->head ) > (intptr_t)T ) {
acquire_task_pool();
H0 = __TBB_load_relaxed( my_arena_slot->head );
if ( (intptr_t)H0 > (intptr_t)T ) {
__TBB_ASSERT( H0 == __TBB_load_relaxed( my_arena_slot->head )
&& T == __TBB_load_relaxed( my_arena_slot->tail )
&& H0 == T + 1, "victim/thief arbitration algorithm failure" );
reset_task_pool_and_leave();
task_pool_empty = true;
break;
} else if ( H0 == T ) {
reset_task_pool_and_leave();
task_pool_empty = true;
} else {
release_task_pool();
}
}
__TBB_control_consistency_helper(); 
#if __TBB_TASK_ISOLATION
result = get_task( T, isolation, tasks_omitted );
if ( result ) {
poison_pointer( my_arena_slot->task_pool_ptr[T] );
break;
} else if ( !tasks_omitted ) {
poison_pointer( my_arena_slot->task_pool_ptr[T] );
__TBB_ASSERT( T0 == T+1, NULL );
T0 = T;
}
#else
result = get_task( T );
#endif 
} while ( !result && !task_pool_empty );

#if __TBB_TASK_ISOLATION
if ( tasks_omitted ) {
if ( task_pool_empty ) {
__TBB_ASSERT( is_quiescent_local_task_pool_reset(), NULL );
if ( result ) {
__TBB_ASSERT( H0 == T, NULL );
++H0;
}
__TBB_ASSERT( H0 <= T0, NULL );
if ( H0 < T0 ) {
__TBB_store_relaxed( my_arena_slot->head, H0 );
__TBB_store_relaxed( my_arena_slot->tail, T0 );
publish_task_pool();
my_arena->advertise_new_work<arena::wakeup>();
}
} else {
__TBB_ASSERT( is_task_pool_published(), NULL );
__TBB_ASSERT( result, NULL );
my_arena_slot->task_pool_ptr[T] = NULL;
__TBB_store_with_release( my_arena_slot->tail, T0 );
my_arena->advertise_new_work<arena::wakeup>();
}

if ( my_innermost_running_task == result ) {
assert_task_valid( result );
result->note_affinity( my_affinity_id );
}
}
#endif 
__TBB_ASSERT( (intptr_t)__TBB_load_relaxed( my_arena_slot->tail ) >= 0, NULL );
__TBB_ASSERT( result || __TBB_ISOLATION_EXPR( tasks_omitted || ) is_quiescent_local_task_pool_reset(), NULL );
return result;
} 

task* generic_scheduler::steal_task( __TBB_ISOLATION_EXPR(isolation_tag isolation) ) {
size_t k = my_random.get() % (my_arena->my_limit-1);
arena_slot* victim = &my_arena->my_slots[k];
if( k >= my_arena_index )
++victim;               
task **pool = victim->task_pool;
task *t = NULL;
if( pool == EmptyTaskPool || !(t = steal_task_from( __TBB_ISOLATION_ARG(*victim, isolation) )) )
return NULL;
if( is_proxy(*t) ) {
task_proxy &tp = *(task_proxy*)t;
t = tp.extract_task<task_proxy::pool_bit>();
if ( !t ) {
free_task<no_cache_small_task>(tp);
return NULL;
}
GATHER_STATISTIC( ++my_counters.proxies_stolen );
}
t->prefix().extra_state |= es_task_is_stolen;
if( is_version_3_task(*t) ) {
my_innermost_running_task = t;
t->prefix().owner = this;
t->note_affinity( my_affinity_id );
}
GATHER_STATISTIC( ++my_counters.steals_committed );
return t;
}

task* generic_scheduler::steal_task_from( __TBB_ISOLATION_ARG( arena_slot& victim_slot, isolation_tag isolation ) ) {
task** victim_pool = lock_task_pool( &victim_slot );
if ( !victim_pool )
return NULL;
task* result = NULL;
size_t H = __TBB_load_relaxed(victim_slot.head); 
size_t H0 = H;
bool tasks_omitted = false;
do {
__TBB_store_relaxed( victim_slot.head, ++H );
atomic_fence();
if ( (intptr_t)H > (intptr_t)__TBB_load_relaxed( victim_slot.tail ) ) {
GATHER_STATISTIC( ++my_counters.thief_backoffs );
__TBB_store_relaxed( victim_slot.head,  H0 );
__TBB_ASSERT( !result, NULL );
goto unlock;
}
__TBB_control_consistency_helper(); 
result = victim_pool[H-1];
__TBB_ASSERT( !is_poisoned( result ), NULL );

if ( result ) {
__TBB_ISOLATION_EXPR( if ( isolation == no_isolation || isolation == result->prefix().isolation ) )
{
if ( !is_proxy( *result ) )
break;
task_proxy& tp = *static_cast<task_proxy*>(result);
if ( !(task_proxy::is_shared( tp.task_and_tag ) && tp.outbox->recipient_is_idle()) )
break;
GATHER_STATISTIC( ++my_counters.proxies_bypassed );
}
result = NULL;
tasks_omitted = true;
} else if ( !tasks_omitted ) {
__TBB_ASSERT( H0 == H-1, NULL );
poison_pointer( victim_pool[H0] );
H0 = H;
}
} while ( !result );
__TBB_ASSERT( result, NULL );

ITT_NOTIFY( sync_acquired, (void*)((uintptr_t)&victim_slot+sizeof( uintptr_t )) );
poison_pointer( victim_pool[H-1] );
if ( tasks_omitted ) {
victim_pool[H-1] = NULL;
__TBB_store_relaxed( victim_slot.head,  H0 );
}
unlock:
unlock_task_pool( &victim_slot, victim_pool );
#if __TBB_PREFETCHING
__TBB_cl_evict(&victim_slot.head);
__TBB_cl_evict(&victim_slot.tail);
#endif
if ( tasks_omitted )
my_arena->advertise_new_work<arena::wakeup>();
return result;
}

#if __TBB_PREVIEW_CRITICAL_TASKS
task* generic_scheduler::get_critical_task( __TBB_ISOLATION_EXPR(isolation_tag isolation) ) {
__TBB_ASSERT( my_arena && my_arena_slot, "Must be attached to arena" );
if( my_arena->my_critical_task_stream.empty(0) )
return NULL;
task* critical_task = NULL;
unsigned& start_lane = my_arena_slot->hint_for_critical;
#if __TBB_TASK_ISOLATION
if( isolation != no_isolation ) {
critical_task = my_arena->my_critical_task_stream.pop_specific( 0, start_lane, isolation );
} else
#endif
if( !my_properties.has_taken_critical_task ) {
critical_task = my_arena->my_critical_task_stream.pop( 0, preceding_lane_selector(start_lane) );
}
return critical_task;
}
#endif

task* generic_scheduler::get_mailbox_task( __TBB_ISOLATION_EXPR( isolation_tag isolation ) ) {
__TBB_ASSERT( my_affinity_id>0, "not in arena" );
while ( task_proxy* const tp = my_inbox.pop( __TBB_ISOLATION_EXPR( isolation ) ) ) {
if ( task* result = tp->extract_task<task_proxy::mailbox_bit>() ) {
ITT_NOTIFY( sync_acquired, my_inbox.outbox() );
result->prefix().extra_state |= es_task_is_stolen;
return result;
}
free_task<no_cache_small_task>(*tp);
}
return NULL;
}

inline void generic_scheduler::publish_task_pool() {
__TBB_ASSERT ( my_arena, "no arena: initialization not completed?" );
__TBB_ASSERT ( my_arena_index < my_arena->my_num_slots, "arena slot index is out-of-bound" );
__TBB_ASSERT ( my_arena_slot == &my_arena->my_slots[my_arena_index], NULL);
__TBB_ASSERT ( my_arena_slot->task_pool == EmptyTaskPool, "someone else grabbed my arena slot?" );
__TBB_ASSERT ( __TBB_load_relaxed(my_arena_slot->head) < __TBB_load_relaxed(my_arena_slot->tail),
"entering arena without tasks to share" );
ITT_NOTIFY(sync_releasing, my_arena_slot);
__TBB_store_with_release( my_arena_slot->task_pool, my_arena_slot->task_pool_ptr );
}

inline void generic_scheduler::leave_task_pool() {
__TBB_ASSERT( is_task_pool_published(), "Not in arena" );
__TBB_ASSERT( &my_arena->my_slots[my_arena_index] == my_arena_slot, "arena slot and slot index mismatch" );
__TBB_ASSERT ( my_arena_slot->task_pool == LockedTaskPool, "Task pool must be locked when leaving arena" );
__TBB_ASSERT ( is_quiescent_local_task_pool_empty(), "Cannot leave arena when the task pool is not empty" );
ITT_NOTIFY(sync_releasing, &my_arena->my_slots[my_arena_index]);
__TBB_store_relaxed( my_arena_slot->task_pool, EmptyTaskPool );
}

generic_scheduler* generic_scheduler::create_worker( market& m, size_t index, bool genuine ) {
generic_scheduler* s = allocate_scheduler( m, genuine );
__TBB_ASSERT(!genuine || index, "workers should have index > 0");
s->my_arena_index = index; 
s->my_dummy_task->prefix().ref_count = 2;
s->my_properties.type = scheduler_properties::worker;
if (genuine)
s->init_stack_info();
governor::sign_on(s);
return s;
}

generic_scheduler* generic_scheduler::create_master( arena* a ) {
generic_scheduler* s = allocate_scheduler( market::global_market(false),  true );
__TBB_ASSERT( !s->my_arena, NULL );
__TBB_ASSERT( s->my_market, NULL );
task& t = *s->my_dummy_task;
s->my_properties.type = scheduler_properties::master;
t.prefix().ref_count = 1;
#if __TBB_TASK_GROUP_CONTEXT
t.prefix().context = new ( NFS_Allocate(1, sizeof(task_group_context), NULL) )
task_group_context( task_group_context::isolated, task_group_context::default_traits );
#if __TBB_FP_CONTEXT
s->default_context()->capture_fp_settings();
#endif
s->init_stack_info();
context_state_propagation_mutex_type::scoped_lock lock(the_context_state_propagation_mutex);
s->my_market->my_masters.push_front( *s );
lock.release();
#endif 
if( a ) {
s->attach_arena( a, 0, true );
s->my_arena_slot->my_scheduler = s;
#if __TBB_TASK_GROUP_CONTEXT
a->my_default_ctx = s->default_context(); 
#endif
}
__TBB_ASSERT( s->my_arena_index == 0, "Master thread must occupy the first slot in its arena" );
governor::sign_on(s);

#if _WIN32||_WIN64
s->my_market->register_master( s->master_exec_resource );
#endif 
#if __TBB_ARENA_OBSERVER
__TBB_ASSERT( !a || a->my_observers.empty(), "Just created arena cannot have any observers associated with it" );
#endif
#if __TBB_SCHEDULER_OBSERVER
the_global_observer_list.notify_entry_observers( s->my_last_global_observer, false );
#endif 
return s;
}

void generic_scheduler::cleanup_worker( void* arg, bool worker ) {
generic_scheduler& s = *(generic_scheduler*)arg;
__TBB_ASSERT( !s.my_arena_slot, "cleaning up attached worker" );
#if __TBB_SCHEDULER_OBSERVER
if ( worker ) 
the_global_observer_list.notify_exit_observers( s.my_last_global_observer, true );
#endif 
s.cleanup_scheduler();
}

bool generic_scheduler::cleanup_master( bool blocking_terminate ) {
arena* const a = my_arena;
market * const m = my_market;
__TBB_ASSERT( my_market, NULL );
if( a && is_task_pool_published() ) {
acquire_task_pool();
if ( my_arena_slot->task_pool == EmptyTaskPool ||
__TBB_load_relaxed(my_arena_slot->head) >= __TBB_load_relaxed(my_arena_slot->tail) )
{
leave_task_pool();
}
else {
release_task_pool();
__TBB_ASSERT ( governor::is_set(this), "TLS slot is cleared before the task pool cleanup" );
my_dummy_task->set_ref_count(2);
local_wait_for_all( *my_dummy_task, NULL );
__TBB_ASSERT( !is_task_pool_published(), NULL );
__TBB_ASSERT ( governor::is_set(this), "Other thread reused our TLS key during the task pool cleanup" );
}
}
#if __TBB_ARENA_OBSERVER
if( a )
a->my_observers.notify_exit_observers( my_last_local_observer, false );
#endif
#if __TBB_SCHEDULER_OBSERVER
the_global_observer_list.notify_exit_observers( my_last_global_observer, false );
#endif 
#if _WIN32||_WIN64
m->unregister_master( master_exec_resource );
#endif 
if( a ) {
__TBB_ASSERT(a->my_slots+0 == my_arena_slot, NULL);
#if __TBB_STATISTICS
*my_arena_slot->my_counters += my_counters;
#endif 
__TBB_store_with_release(my_arena_slot->my_scheduler, (generic_scheduler*)NULL);
}
#if __TBB_TASK_GROUP_CONTEXT
else { 
default_context()->~task_group_context();
NFS_Free(default_context());
}
context_state_propagation_mutex_type::scoped_lock lock(the_context_state_propagation_mutex);
my_market->my_masters.remove( *this );
lock.release();
#endif 
my_arena_slot = NULL; 
cleanup_scheduler(); 

if( a )
a->on_thread_leaving<arena::ref_external>();
return m->release(  a != NULL, blocking_terminate );
}

} 
} 


