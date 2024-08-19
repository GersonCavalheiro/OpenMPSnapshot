

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_task_scheduler_init_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_task_scheduler_init_H
#pragma message("TBB Warning: tbb/task_scheduler_init.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_task_scheduler_init_H
#define __TBB_task_scheduler_init_H

#define __TBB_task_scheduler_init_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"
#include "limits.h"
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
#include <new> 
#endif

namespace tbb {

typedef std::size_t stack_size_type;

namespace internal {

class scheduler;
} 


class __TBB_DEPRECATED_IN_VERBOSE_MODE task_scheduler_init: internal::no_copy {
enum ExceptionPropagationMode {
propagation_mode_exact = 1u,
propagation_mode_captured = 2u,
propagation_mode_mask = propagation_mode_exact | propagation_mode_captured
};


internal::scheduler* my_scheduler;

bool internal_terminate( bool blocking );
#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
bool __TBB_EXPORTED_METHOD internal_blocking_terminate( bool throwing );
#endif
public:

static const int automatic = -1;

static const int deferred = -2;


void __TBB_EXPORTED_METHOD initialize( int number_of_threads=automatic );


void __TBB_EXPORTED_METHOD initialize( int number_of_threads, stack_size_type thread_stack_size );

void __TBB_EXPORTED_METHOD terminate();

#if __TBB_SUPPORTS_WORKERS_WAITING_IN_TERMINATE
#if TBB_USE_EXCEPTIONS
void blocking_terminate() {
internal_blocking_terminate( true );
}
#endif
bool blocking_terminate(const std::nothrow_t&) __TBB_NOEXCEPT(true) {
return internal_blocking_terminate( false );
}
#endif 

task_scheduler_init( int number_of_threads=automatic, stack_size_type thread_stack_size=0 ) : my_scheduler(NULL)
{
__TBB_ASSERT( !(thread_stack_size & propagation_mode_mask), "Requested stack size is not aligned" );
#if TBB_USE_EXCEPTIONS
thread_stack_size |= TBB_USE_CAPTURED_EXCEPTION ? propagation_mode_captured : propagation_mode_exact;
#endif 
initialize( number_of_threads, thread_stack_size );
}

~task_scheduler_init() {
if( my_scheduler )
terminate();
internal::poison_pointer( my_scheduler );
}

static int __TBB_EXPORTED_FUNC default_num_threads ();

bool is_active() const { return my_scheduler != NULL; }
};

} 

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_task_scheduler_init_H_include_area

#endif 
