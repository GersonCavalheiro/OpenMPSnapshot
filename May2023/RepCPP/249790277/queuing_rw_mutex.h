

#ifndef __TBB_queuing_rw_mutex_H
#define __TBB_queuing_rw_mutex_H

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


class queuing_rw_mutex : internal::mutex_copy_deprecated_and_disabled {
public:
queuing_rw_mutex() {
q_tail = NULL;
#if TBB_USE_THREADING_TOOLS
internal_construct();
#endif
}

~queuing_rw_mutex() {
#if TBB_USE_ASSERT
__TBB_ASSERT( !q_tail, "destruction of an acquired mutex");
#endif
}


class scoped_lock: internal::no_copy {
void initialize() {
my_mutex = NULL;
#if TBB_USE_ASSERT
my_state = 0xFF; 
internal::poison_pointer(my_next);
internal::poison_pointer(my_prev);
#endif 
}

public:

scoped_lock() {initialize();}

scoped_lock( queuing_rw_mutex& m, bool write=true ) {
initialize();
acquire(m,write);
}

~scoped_lock() {
if( my_mutex ) release();
}

void acquire( queuing_rw_mutex& m, bool write=true );

bool try_acquire( queuing_rw_mutex& m, bool write=true );

void release();


bool upgrade_to_writer();

bool downgrade_to_reader();

private:
queuing_rw_mutex* my_mutex;

scoped_lock *__TBB_atomic my_prev, *__TBB_atomic my_next;

typedef unsigned char state_t;

atomic<state_t> my_state;


unsigned char __TBB_atomic my_going;

unsigned char my_internal_lock;

void acquire_internal_lock();


bool try_acquire_internal_lock();

void release_internal_lock();

void wait_for_release_of_internal_lock();

void unblock_or_wait_on_internal_lock( uintptr_t );
};

void __TBB_EXPORTED_METHOD internal_construct();

static const bool is_rw_mutex = true;
static const bool is_recursive_mutex = false;
static const bool is_fair_mutex = true;

private:
atomic<scoped_lock*> q_tail;

};

__TBB_DEFINE_PROFILING_SET_NAME(queuing_rw_mutex)

} 

#endif 
