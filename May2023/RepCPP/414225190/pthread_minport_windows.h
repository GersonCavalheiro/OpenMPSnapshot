



#ifndef FF_MINPORT_WIN_H
#define FF_MINPORT_WIN_H


#pragma once
#define NOMINMAX
#include <Windows.h>
#include <WinBase.h>
#include <process.h>
#include <intrin.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>



#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#define RESTRICT __declspec()


#define PTHREAD_CANCEL_ENABLE        0x01  
#define PTHREAD_CANCEL_DISABLE       0x00  
#define PTHREAD_CANCEL_DEFERRED      0x02  

typedef HANDLE pthread_t;
typedef struct _opaque_pthread_attr_t { long __sig; } pthread_attr_t;
typedef struct _opaque_pthread_mutexattr_t { long __sig; } pthread_mutexattr_t;
typedef struct _opaque_pthread_condattr_t {long __sig; } pthread_condattr_t;

typedef DWORD pthread_key_t;



typedef CRITICAL_SECTION pthread_mutex_t;

INLINE int pthread_create(pthread_t RESTRICT * thread,
const pthread_attr_t RESTRICT * attr, 
void *(*start_routine)(void *), 
void RESTRICT * arg) {
thread[0] = (HANDLE)_beginthreadex(0, 0, (unsigned(__stdcall*)(void*))start_routine, arg, 0, 0);
return(0); 
}


INLINE int pthread_join(pthread_t thread, void **value_ptr) {
LPDWORD exitcode = 0;
WaitForSingleObject(thread, INFINITE);
if (value_ptr)  {
GetExitCodeThread(thread,exitcode);
*value_ptr = exitcode;
}
CloseHandle(thread);
return(0); 
}

INLINE void pthread_exit(void *value_ptr) {
if (value_ptr)
ExitThread(*((DWORD *) value_ptr));
else 
ExitThread(0);
}

INLINE int pthread_attr_init(pthread_attr_t *attr) {
return(0);
}

INLINE int pthread_attr_destroy(pthread_attr_t *attr) {
return(0);
}

INLINE int pthread_setcancelstate(int state, int *oldstate) {
return(0);
}


INLINE int pthread_mutex_init(pthread_mutex_t  RESTRICT * mutex,
const pthread_mutexattr_t RESTRICT  * attr) {
if (attr) return(EINVAL);
InitializeCriticalSectionAndSpinCount(mutex, 1500 );	
return (0); 
}

INLINE int pthread_mutex_lock(pthread_mutex_t *mutex) {
EnterCriticalSection(mutex);
return(0); 
}

INLINE int pthread_mutex_unlock(pthread_mutex_t *mutex) {
LeaveCriticalSection(mutex);
return(0); 
}

INLINE int pthread_mutex_destroy(pthread_mutex_t *mutex) {
DeleteCriticalSection(mutex);
return(0); 
}


INLINE  pthread_t pthread_self(void) {
return(GetCurrentThread());
}

INLINE int pthread_key_create(pthread_key_t *key, void (*destructor)(void *)) {
*key = TlsAlloc();
return 0;
}

INLINE int pthread_key_delete(pthread_key_t key) {
TlsFree(key);
return 0;
}

INLINE  int pthread_setspecific(pthread_key_t key, const void *value) {
TlsSetValue(key, (LPVOID) value);
return (0);
}

INLINE void * pthread_getspecific(pthread_key_t key) {
return(TlsGetValue(key));
}

#ifndef _FF_WIN_XP
typedef CONDITION_VARIABLE pthread_cond_t;
#include <WinBase.h>
INLINE int pthread_cond_init(pthread_cond_t  RESTRICT * cond,
const pthread_condattr_t  RESTRICT * attr) {
if (attr) return(EINVAL);
InitializeConditionVariable(cond);	
return (0);	 
}

INLINE int pthread_cond_signal(pthread_cond_t *cond) {
WakeConditionVariable(cond);
return(0); 
}

INLINE int pthread_cond_broadcast(pthread_cond_t *cond) {
WakeAllConditionVariable(cond);
return(0);
}

INLINE int pthread_cond_wait(pthread_cond_t  RESTRICT * cond,
pthread_mutex_t  RESTRICT * mutex) {
SleepConditionVariableCS(cond, mutex, INFINITE);
return(0); 
}

INLINE int pthread_cond_destroy(pthread_cond_t *cond) {
return (0); 
}




#else 

typedef struct
{
int waiters_count_;

CRITICAL_SECTION waiters_count_lock_;

int release_count_;

int wait_generation_count_;

HANDLE event_;
} pthread_cond_t;

INLINE int pthread_cond_init(pthread_cond_t  RESTRICT * cv,
const pthread_condattr_t  RESTRICT * attr) {
cv->waiters_count_ = 0;
cv->wait_generation_count_ = 0;
cv->release_count_ = 0;

cv->event_ = CreateEvent (NULL,  
TRUE,  
FALSE, 
NULL); 

pthread_mutex_init(&cv->waiters_count_lock_,NULL);
return 0;
}

INLINE int pthread_cond_wait(pthread_cond_t  RESTRICT * cv,
pthread_mutex_t  RESTRICT * external_mutex) {
EnterCriticalSection (&cv->waiters_count_lock_);

cv->waiters_count_++;

int my_generation = cv->wait_generation_count_;

LeaveCriticalSection (&cv->waiters_count_lock_);
LeaveCriticalSection (external_mutex);

for (;;) {
WaitForSingleObject (cv->event_, INFINITE);

EnterCriticalSection (&cv->waiters_count_lock_);
int wait_done = cv->release_count_ > 0
&& cv->wait_generation_count_ != my_generation;
LeaveCriticalSection (&cv->waiters_count_lock_);

if (wait_done)
break;
}

EnterCriticalSection (external_mutex);
EnterCriticalSection (&cv->waiters_count_lock_);
cv->waiters_count_--;
cv->release_count_--;
int last_waiter = cv->release_count_ == 0;
LeaveCriticalSection (&cv->waiters_count_lock_);

if (last_waiter)
ResetEvent (cv->event_);
return 0;
}

INLINE int pthread_cond_signal(pthread_cond_t *cv) {
EnterCriticalSection (&cv->waiters_count_lock_);
if (cv->waiters_count_ > cv->release_count_) {
SetEvent (cv->event_); 
cv->release_count_++;
cv->wait_generation_count_++;
}
LeaveCriticalSection (&cv->waiters_count_lock_);
return 0;
}

INLINE int pthread_cond_broadcast(pthread_cond_t *cv) {
EnterCriticalSection (&cv->waiters_count_lock_);
if (cv->waiters_count_ > 0) {  
SetEvent (cv->event_);
cv->release_count_ = cv->waiters_count_;

cv->wait_generation_count_++;
}
LeaveCriticalSection (&cv->waiters_count_lock_);
return 0;
}

INLINE int pthread_cond_destroy(pthread_cond_t *cond) {
return (0); 
}

#endif

#endif 
