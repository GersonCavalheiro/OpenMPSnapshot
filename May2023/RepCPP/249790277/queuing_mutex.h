

#ifndef __TBB_queuing_mutex_H
#define __TBB_queuing_mutex_H

#include "tbb_config.h"

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <cstring>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#include "atomic.h"
#include "tbb_profiling.h"

namespace tbb {


class queuing_mutex : internal::mutex_copy_deprecated_and_disabled {
public:
queuing_mutex() {
q_tail = NULL;
#if TBB_USE_THREADING_TOOLS
internal_construct();
#endif
}


class scoped_lock: internal::no_copy {
void initialize() {
mutex = NULL;
#if TBB_USE_ASSERT
internal::poison_pointer(next);
#endif 
}

public:

scoped_lock() {initialize();}

scoped_lock( queuing_mutex& m ) {
initialize();
acquire(m);
}

~scoped_lock() {
if( mutex ) release();
}

void __TBB_EXPORTED_METHOD acquire( queuing_mutex& m );

bool __TBB_EXPORTED_METHOD try_acquire( queuing_mutex& m );

void __TBB_EXPORTED_METHOD release();

private:
queuing_mutex* mutex;

scoped_lock *next;


uintptr_t going;
};

void __TBB_EXPORTED_METHOD internal_construct();

static const bool is_rw_mutex = false;
static const bool is_recursive_mutex = false;
static const bool is_fair_mutex = true;

private:
atomic<scoped_lock*> q_tail;

};

__TBB_DEFINE_PROFILING_SET_NAME(queuing_mutex)

} 

#endif 
