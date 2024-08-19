

#ifndef _TBB_scheduler_common_H
#define _TBB_scheduler_common_H

#include "tbb/tbb_machine.h"
#include "tbb/cache_aligned_allocator.h"

#include <string.h>  

#include "tbb_statistics.h"

#if TBB_USE_ASSERT > 1
#include <stdio.h>
#endif 


#ifndef private
#define private public
#define undef_private
#endif

#include "tbb/task.h"
#include "tbb/tbb_exception.h"

#ifdef undef_private
#undef private
#endif

#ifndef __TBB_SCHEDULER_MUTEX_TYPE
#define __TBB_SCHEDULER_MUTEX_TYPE tbb::spin_mutex
#endif
#include "tbb/spin_mutex.h"

#if __TBB_TASK_GROUP_CONTEXT
#define __TBB_CONTEXT_ARG1(context) context
#define __TBB_CONTEXT_ARG(arg1, context) arg1, context
#else 
#define __TBB_CONTEXT_ARG1(context)
#define __TBB_CONTEXT_ARG(arg1, context) arg1
#endif 

#if __TBB_TASK_ISOLATION
#define __TBB_ISOLATION_EXPR(isolation) isolation
#define __TBB_ISOLATION_ARG(arg1, isolation) arg1, isolation
#else
#define __TBB_ISOLATION_EXPR(isolation)
#define __TBB_ISOLATION_ARG(arg1, isolation) arg1
#endif 


#if DO_TBB_TRACE
#include <cstdio>
#define TBB_TRACE(x) ((void)std::printf x)
#else
#define TBB_TRACE(x) ((void)(0))
#endif 

#if !__TBB_CPU_CTL_ENV_PRESENT
#include <fenv.h>
#endif

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (disable: 4100 4127 4312 4244 4267 4706)
#endif

namespace tbb {
namespace interface7 {
namespace internal {
class task_arena_base;
class delegated_task;
class wait_task;
}}
namespace internal {
using namespace interface7::internal;

class arena;
template<typename SchedulerTraits> class custom_scheduler;
class generic_scheduler;
class governor;
class mail_outbox;
class market;
class observer_proxy;
class task_scheduler_observer_v3;

#if __TBB_TASK_PRIORITY
static const intptr_t num_priority_levels = 3;
static const intptr_t normalized_normal_priority = (num_priority_levels - 1) / 2;

inline intptr_t normalize_priority ( priority_t p ) {
return intptr_t(p - priority_low) / priority_stride_v4;
}

static const priority_t priority_from_normalized_rep[num_priority_levels] = {
priority_low, priority_normal, priority_high
};

inline void assert_priority_valid ( intptr_t p ) {
__TBB_ASSERT_EX( p >= 0 && p < num_priority_levels, NULL );
}

inline intptr_t& priority ( task& t ) {
return t.prefix().context->my_priority;
}
#else 
static const intptr_t num_priority_levels = 1;
#endif 

typedef __TBB_SCHEDULER_MUTEX_TYPE scheduler_mutex_type;

#if __TBB_TASK_GROUP_CONTEXT

extern uintptr_t the_context_state_propagation_epoch;


typedef scheduler_mutex_type context_state_propagation_mutex_type;
extern context_state_propagation_mutex_type the_context_state_propagation_mutex;
#endif 

const size_t task_alignment = 32;


const size_t task_prefix_reservation_size = ((sizeof(internal::task_prefix)-1)/task_alignment+1)*task_alignment;

enum task_extra_state {
es_version_1_task = 0,
es_version_3_task = 1,
#if __TBB_PREVIEW_CRITICAL_TASKS
es_task_critical = 0x8,
#endif
es_task_enqueued = 0x10,
es_task_proxy = 0x20,
es_ref_count_active = 0x40,
es_task_is_stolen = 0x80
};

inline void reset_extra_state ( task *t ) {
t->prefix().extra_state &= ~(es_task_is_stolen | es_task_enqueued);
}

enum free_task_hint {
no_hint=0,
local_task=1,

small_task=2,

small_local_task=3,
no_cache = 4,
no_cache_small_task = no_cache | small_task
};


#if TBB_USE_ASSERT

static const uintptr_t venom = tbb::internal::select_size_t_constant<0xDEADBEEFU,0xDDEEAADDDEADBEEFULL>::value;

template <typename T>
void poison_value ( T& val ) { val = * punned_cast<T*>(&venom); }


inline bool is_alive( uintptr_t v ) { return v != venom; }


inline void assert_task_valid( const task* task ) {
__TBB_ASSERT( task!=NULL, NULL );
__TBB_ASSERT( !is_poisoned(&task), NULL );
__TBB_ASSERT( (uintptr_t)task % task_alignment == 0, "misaligned task" );
__TBB_ASSERT( task->prefix().ref_count >= 0, NULL);
#if __TBB_RECYCLE_TO_ENQUEUE
__TBB_ASSERT( (unsigned)task->state()<=(unsigned)task::to_enqueue, "corrupt task (invalid state)" );
#else
__TBB_ASSERT( (unsigned)task->state()<=(unsigned)task::recycle, "corrupt task (invalid state)" );
#endif
}

#else 


#define poison_value(g) ((void)0)

inline void assert_task_valid( const task* ) {}

#endif 


#if __TBB_TASK_GROUP_CONTEXT
inline bool ConcurrentWaitsEnabled ( task& t ) {
return (t.prefix().context->my_version_and_traits & task_group_context::concurrent_wait) != 0;
}

inline bool CancellationInfoPresent ( task& t ) {
return t.prefix().context->my_cancellation_requested != 0;
}

#if TBB_USE_CAPTURED_EXCEPTION
inline tbb_exception* TbbCurrentException( task_group_context*, tbb_exception* src) { return src->move(); }
inline tbb_exception* TbbCurrentException( task_group_context* c, captured_exception* src) {
if( c->my_version_and_traits & task_group_context::exact_exception )
runtime_warning( "Exact exception propagation is requested by application but the linked library is built without support for it");
return src;
}
#define TbbRethrowException(TbbCapturedException) (TbbCapturedException)->throw_self()
#else
#define TbbCurrentException(context, TbbCapturedException) \
context->my_version_and_traits & task_group_context::exact_exception    \
? tbb_exception_ptr::allocate()    \
: tbb_exception_ptr::allocate( *(TbbCapturedException) );
#define TbbRethrowException(TbbCapturedException) \
{ \
if( governor::rethrow_exception_broken() ) fix_broken_rethrow(); \
(TbbCapturedException)->throw_self(); \
}
#endif 

#define TbbRegisterCurrentException(context, TbbCapturedException) \
if ( context->cancel_group_execution() ) {  \
\
context->my_exception = TbbCurrentException( context, TbbCapturedException ); \
}

#define TbbCatchAll(context)  \
catch ( tbb_exception& exc ) {  \
TbbRegisterCurrentException( context, &exc );   \
} catch ( std::exception& exc ) {   \
TbbRegisterCurrentException( context, captured_exception::allocate(typeid(exc).name(), exc.what()) ); \
} catch ( ... ) {   \
TbbRegisterCurrentException( context, captured_exception::allocate("...", "Unidentified exception") );\
}

#else 

inline bool ConcurrentWaitsEnabled ( task& t ) { return false; }

#endif 

inline void prolonged_pause() {
#if defined(__TBB_time_stamp) && !__TBB_STEALING_PAUSE
machine_tsc_t prev = __TBB_time_stamp();
const machine_tsc_t finish = prev + 1000;
atomic_backoff backoff;
do {
backoff.bounded_pause();
machine_tsc_t curr = __TBB_time_stamp();
if ( curr <= prev )
break;
prev = curr;
} while ( prev < finish );
#else
#ifdef __TBB_STEALING_PAUSE
static const long PauseTime = __TBB_STEALING_PAUSE;
#elif __TBB_ipf
static const long PauseTime = 1500;
#else
static const long PauseTime = 80;
#endif
__TBB_Pause(PauseTime);
#endif
}

struct arena_slot_line1 {

generic_scheduler* my_scheduler;


task* *__TBB_atomic task_pool;


__TBB_atomic size_t head;

#if __TBB_PREVIEW_RESUMABLE_TASKS
tbb::atomic<bool>* my_scheduler_is_recalled;
#endif
};

struct arena_slot_line2 {

unsigned hint_for_pop;

#if __TBB_PREVIEW_CRITICAL_TASKS
unsigned hint_for_critical;
#endif


__TBB_atomic size_t tail;

size_t my_task_pool_size;

task* *__TBB_atomic task_pool_ptr;

#if __TBB_STATISTICS
statistics_counters *my_counters;
#endif 
};

struct arena_slot : padded<arena_slot_line1>, padded<arena_slot_line2> {
#if TBB_USE_ASSERT
void fill_with_canary_pattern ( size_t first, size_t last ) {
for ( size_t i = first; i < last; ++i )
poison_pointer(task_pool_ptr[i]);
}
#else
void fill_with_canary_pattern ( size_t, size_t ) {}
#endif 

void allocate_task_pool( size_t n ) {
size_t byte_size = ((n * sizeof(task*) + NFS_MaxLineSize - 1) / NFS_MaxLineSize) * NFS_MaxLineSize;
my_task_pool_size = byte_size / sizeof(task*);
task_pool_ptr = (task**)NFS_Allocate( 1, byte_size, NULL );
fill_with_canary_pattern( 0, my_task_pool_size );
}

void free_task_pool( ) {
if( task_pool_ptr ) {
__TBB_ASSERT( my_task_pool_size, NULL);
NFS_Free( task_pool_ptr );
task_pool_ptr = NULL;
my_task_pool_size = 0;
}
}
};

#if !__TBB_CPU_CTL_ENV_PRESENT
class cpu_ctl_env {
fenv_t *my_fenv_ptr;
public:
cpu_ctl_env() : my_fenv_ptr(NULL) {}
~cpu_ctl_env() {
if ( my_fenv_ptr )
tbb::internal::NFS_Free( (void*)my_fenv_ptr );
}
cpu_ctl_env( const cpu_ctl_env &src ) : my_fenv_ptr(NULL) {
*this = src;
}
cpu_ctl_env& operator=( const cpu_ctl_env &src ) {
__TBB_ASSERT( src.my_fenv_ptr, NULL );
if ( !my_fenv_ptr )
my_fenv_ptr = (fenv_t*)tbb::internal::NFS_Allocate(1, sizeof(fenv_t), NULL);
*my_fenv_ptr = *src.my_fenv_ptr;
return *this;
}
bool operator!=( const cpu_ctl_env &ctl ) const {
__TBB_ASSERT( my_fenv_ptr, "cpu_ctl_env is not initialized." );
__TBB_ASSERT( ctl.my_fenv_ptr, "cpu_ctl_env is not initialized." );
return memcmp( (void*)my_fenv_ptr, (void*)ctl.my_fenv_ptr, sizeof(fenv_t) );
}
void get_env () {
if ( !my_fenv_ptr )
my_fenv_ptr = (fenv_t*)tbb::internal::NFS_Allocate(1, sizeof(fenv_t), NULL);
fegetenv( my_fenv_ptr );
}
const cpu_ctl_env& set_env () const {
__TBB_ASSERT( my_fenv_ptr, "cpu_ctl_env is not initialized." );
fesetenv( my_fenv_ptr );
return *this;
}
};
#endif 

} 
} 

#endif 
