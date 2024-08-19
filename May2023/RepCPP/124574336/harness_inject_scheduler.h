


#ifndef harness_inject_scheduler_H
#define harness_inject_scheduler_H

#if HARNESS_DEFINE_PRIVATE_PUBLIC
#include <string> 
#include <algorithm> 
#define private public
#define protected public
#endif

#define __TBB_NO_IMPLICIT_LINKAGE 1

#define __TBB_BUILD 1

#undef DO_ITT_NOTIFY

#define __TBB_SOURCE_DIRECTLY_INCLUDED 1
#include "../tbb/tbb_main.cpp"
#include "../tbb/dynamic_link.cpp"
#include "../tbb/tbb_misc_ex.cpp"

#include "../tbb/governor.cpp"
#include "../tbb/market.cpp"
#include "../tbb/arena.cpp"
#include "../tbb/scheduler.cpp"
#include "../tbb/observer_proxy.cpp"
#include "../tbb/task.cpp"
#include "../tbb/task_group_context.cpp"

#include "../tbb/cache_aligned_allocator.cpp"
#include "../tbb/tbb_thread.cpp"
#include "../tbb/mutex.cpp"
#include "../tbb/spin_rw_mutex.cpp"
#include "../tbb/spin_mutex.cpp"
#include "../tbb/private_server.cpp"
#include "../tbb/concurrent_monitor.cpp"
#if _WIN32||_WIN64
#include "../tbb/semaphore.cpp"
#endif
#include "../rml/client/rml_tbb.cpp"

#if HARNESS_USE_RUNTIME_LOADER
#undef HARNESS_USE_RUNTIME_LOADER
#include "harness.h"

int TestMain () {
return Harness::Skipped;
}
#define TestMain TestMainSkipped
#endif

#if HARNESS_DEFINE_PRIVATE_PUBLIC
#undef protected
#undef private
#endif

#endif 
