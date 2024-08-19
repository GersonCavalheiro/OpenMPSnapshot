#pragma once
#ifdef _MSC_VER
#include <windows.h>
enum {
thrd_success = 0,
thrd_nomem = 1,
thrd_timedout = 2,
thrd_busy = 3,
thrd_error = 4
};
typedef HANDLE thrd_t;
typedef LPTHREAD_START_ROUTINE thrd_start_t;
SINLINE int thrd_create(thrd_t* thr, thrd_start_t func, void* arg)
{
DWORD thrdId;
*thr = CreateThread(NULL, 0, func, arg, 0, &thrdId);
if (*thr == NULL)
return thrd_error;
return thrd_success;
}
SINLINE int thrd_equal(thrd_t lhs, thrd_t rhs)
{
return lhs == rhs ? 1 : 0;
}
SINLINE thrd_t thrd_current()
{
return GetCurrentThread();
}
SINLINE int thrd_sleep(const struct timespec* duration, struct timespec* remaining)
{
DWORD millis = (DWORD)((duration->tv_sec * 1000000000 + duration->tv_nsec) / 1000000);
Sleep(millis);
if (remaining != NULL)
{
remaining->tv_sec = 0;
remaining->tv_nsec = 0;
}
return 0;
}
SINLINE void thrd_yield()
{
SwitchToThread();
}
SINLINE _Noreturn void thrd_exit(int res)
{
ExitThread(res);
}
SINLINE int thrd_detach(thrd_t thr)
{
BOOL res = CloseHandle(thr);
if (res == TRUE)
return thrd_success;
return thrd_error;
}
SINLINE int thrd_join(thrd_t thr, int* res)
{
DWORD r = WaitForSingleObject(thr, INFINITE);
if (r != 0)
GetExitCodeThread(thr, res);
if (r != WAIT_OBJECT_0)
return thrd_error;
return thrd_success;
}
enum {
mtx_plain = 0,
mtx_recursive = 1,
mtx_timed = 2
};
typedef HANDLE mtx_t;
SINLINE int mtx_init(mtx_t* mutex, int type)
{
if ((type & mtx_recursive) > 0)
return thrd_error; 
*mutex = CreateMutex(NULL, FALSE, NULL);
return thrd_success;
}
SINLINE int mtx_lock(mtx_t* mutex)
{
DWORD result = WaitForSingleObject(*mutex, INFINITE);
if (result == WAIT_OBJECT_0)
return thrd_success;
return thrd_error;
}
SINLINE int mtx_timedlock(mtx_t* mutex, const struct timespec* time_point)
{
DWORD millis = (DWORD)((time_point->tv_sec * 1000000000 + time_point->tv_nsec) / 1000000);
DWORD result = WaitForSingleObject(*mutex, millis);
if (result == WAIT_OBJECT_0)
return thrd_success;
return thrd_error;
}
SINLINE int mtx_trylock(mtx_t* mutex)
{
DWORD result = WaitForSingleObject(*mutex, 0);
if (result == WAIT_OBJECT_0)
return thrd_success;
else if (result == WAIT_TIMEOUT || result == WAIT_ABANDONED)
return thrd_busy;
return thrd_error;
}
SINLINE int mtx_unlock(mtx_t* mutex)
{
DWORD result = ReleaseMutex(*mutex);
if (result == 0)
return thrd_error;
return thrd_success;
}
SINLINE void mtx_destroy(mtx_t* mutex)
{
CloseHandle(*mutex);
}
#elif !defined(__STDC_NO_THREADS__)
#include <threads.h>
#else
#endif