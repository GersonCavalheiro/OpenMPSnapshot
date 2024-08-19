

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_recursive_mutex_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_recursive_mutex_H
#pragma message("TBB Warning: tbb/recursive_mutex.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_recursive_mutex_H
#define __TBB_recursive_mutex_H

#define __TBB_recursive_mutex_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if _WIN32||_WIN64
#include "machine/windows_api.h"
#else
#include <pthread.h>
#endif 

#include <new>
#include "aligned_space.h"
#include "tbb_stddef.h"
#include "tbb_profiling.h"

namespace tbb {

class __TBB_DEPRECATED_VERBOSE_MSG("tbb::recursive_mutex is deprecated, use std::recursive_mutex")
recursive_mutex : internal::mutex_copy_deprecated_and_disabled {
public:
recursive_mutex() {
#if TBB_USE_ASSERT || TBB_USE_THREADING_TOOLS
internal_construct();
#else
#if _WIN32||_WIN64
InitializeCriticalSectionEx(&impl, 4000, 0);
#else
pthread_mutexattr_t mtx_attr;
int error_code = pthread_mutexattr_init( &mtx_attr );
if( error_code )
tbb::internal::handle_perror(error_code,"recursive_mutex: pthread_mutexattr_init failed");

pthread_mutexattr_settype( &mtx_attr, PTHREAD_MUTEX_RECURSIVE );
error_code = pthread_mutex_init( &impl, &mtx_attr );
if( error_code )
tbb::internal::handle_perror(error_code,"recursive_mutex: pthread_mutex_init failed");

pthread_mutexattr_destroy( &mtx_attr );
#endif 
#endif 
};

~recursive_mutex() {
#if TBB_USE_ASSERT
internal_destroy();
#else
#if _WIN32||_WIN64
DeleteCriticalSection(&impl);
#else
pthread_mutex_destroy(&impl);

#endif 
#endif 
};

class scoped_lock;
friend class scoped_lock;


class scoped_lock: internal::no_copy {
public:
scoped_lock() : my_mutex(NULL) {};

scoped_lock( recursive_mutex& mutex ) {
#if TBB_USE_ASSERT
my_mutex = &mutex;
#endif 
acquire( mutex );
}

~scoped_lock() {
if( my_mutex )
release();
}

void acquire( recursive_mutex& mutex ) {
#if TBB_USE_ASSERT
internal_acquire( mutex );
#else
my_mutex = &mutex;
mutex.lock();
#endif 
}

bool try_acquire( recursive_mutex& mutex ) {
#if TBB_USE_ASSERT
return internal_try_acquire( mutex );
#else
bool result = mutex.try_lock();
if( result )
my_mutex = &mutex;
return result;
#endif 
}

void release() {
#if TBB_USE_ASSERT
internal_release();
#else
my_mutex->unlock();
my_mutex = NULL;
#endif 
}

private:
recursive_mutex* my_mutex;

void __TBB_EXPORTED_METHOD internal_acquire( recursive_mutex& m );

bool __TBB_EXPORTED_METHOD internal_try_acquire( recursive_mutex& m );

void __TBB_EXPORTED_METHOD internal_release();

friend class recursive_mutex;
};

static const bool is_rw_mutex = false;
static const bool is_recursive_mutex = true;
static const bool is_fair_mutex = false;


void lock() {
#if TBB_USE_ASSERT
aligned_space<scoped_lock> tmp;
new(tmp.begin()) scoped_lock(*this);
#else
#if _WIN32||_WIN64
EnterCriticalSection(&impl);
#else
int error_code = pthread_mutex_lock(&impl);
if( error_code )
tbb::internal::handle_perror(error_code,"recursive_mutex: pthread_mutex_lock failed");
#endif 
#endif 
}


bool try_lock() {
#if TBB_USE_ASSERT
aligned_space<scoped_lock> tmp;
return (new(tmp.begin()) scoped_lock)->internal_try_acquire(*this);
#else
#if _WIN32||_WIN64
return TryEnterCriticalSection(&impl)!=0;
#else
return pthread_mutex_trylock(&impl)==0;
#endif 
#endif 
}

void unlock() {
#if TBB_USE_ASSERT
aligned_space<scoped_lock> tmp;
scoped_lock& s = *tmp.begin();
s.my_mutex = this;
s.internal_release();
#else
#if _WIN32||_WIN64
LeaveCriticalSection(&impl);
#else
pthread_mutex_unlock(&impl);
#endif 
#endif 
}

#if _WIN32||_WIN64
typedef LPCRITICAL_SECTION native_handle_type;
#else
typedef pthread_mutex_t* native_handle_type;
#endif
native_handle_type native_handle() { return (native_handle_type) &impl; }

private:
#if _WIN32||_WIN64
CRITICAL_SECTION impl;
enum state_t {
INITIALIZED=0x1234,
DESTROYED=0x789A,
} state;
#else
pthread_mutex_t impl;
#endif 

void __TBB_EXPORTED_METHOD internal_construct();

void __TBB_EXPORTED_METHOD internal_destroy();
};

__TBB_DEFINE_PROFILING_SET_NAME(recursive_mutex)

} 

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_recursive_mutex_H_include_area

#endif 
