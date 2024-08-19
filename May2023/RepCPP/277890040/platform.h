


#ifndef FF_PLATFORM_HPP
#define FF_PLATFORM_HPP

#include <ff/platforms/liblfds.h>


#if defined(__APPLE__)
#include <Availability.h>
#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 1060
#define __FF_HAS_POSIX_MEMALIGN 1
#else
#include <errno.h>
inline static int posix_memalign(void **memptr, size_t alignment, size_t size)
{
if (memptr && (*memptr = malloc(size))) return 0; 
else return (ENOMEM);
}
#endif
#endif




#if defined(_WIN32)
#pragma unmanaged

#define NOMINMAX

#include <ff/platforms/pthread_minport_windows.h>
#define INLINE __forceinline
#define NOINLINE __declspec(noinline)
#define __WIN_ALIGNED_16__ __declspec(align(16))

#define __thread __declspec(thread)


INLINE void WMB() {} 
INLINE void PAUSE() {}

#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

INLINE static int posix_memalign(void **memptr,size_t alignment, size_t sz)
{
*memptr =  _aligned_malloc(sz, alignment);
return(!memptr);
}


INLINE static void posix_memalign_free(void* mem)
{
_aligned_free(mem);
}


#include <string>
typedef unsigned long useconds_t;
#define strtoll _strtoi64

#define sleep(SECS) Sleep(SECS)

INLINE static int usleep(unsigned long microsecs) {
if (microsecs > 100000)

Sleep (microsecs/ 1000);
else
{

static double frequency;
if (frequency == 0)
{
LARGE_INTEGER freq;
if (!QueryPerformanceFrequency (&freq))
{

Sleep (microsecs / 1000);
return 0;
}
frequency = (double) freq.QuadPart / 1000000000.0;
}
long long expected_counter_difference = 1000 * microsecs * (long long) frequency;
int sleep_part = (int) (microsecs) / 1000 - 10;
LARGE_INTEGER before;
QueryPerformanceCounter (&before);
long long expected_counter = before.QuadPart + 
expected_counter_difference;
if (sleep_part > 0)
Sleep (sleep_part);
for (;;)
{
LARGE_INTEGER after;
QueryPerformanceCounter (&after);
if (after.QuadPart >= expected_counter)
break;
}
}
return(0);
}


#define random rand
#define srandom srand
#define getpid _getpid
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif



struct timezone 
{
int  tz_minuteswest; 
int  tz_dsttime;     
};

INLINE static int gettimeofday(struct timeval *tv, struct timezone *tz)
{
FILETIME ft;
unsigned __int64 tmpres = 0;
static int tzflag;

if (NULL != tv)
{
GetSystemTimeAsFileTime(&ft);

tmpres |= ft.dwHighDateTime;
tmpres <<= 32;
tmpres |= ft.dwLowDateTime;


tmpres -= DELTA_EPOCH_IN_MICROSECS; 
tmpres /= 10;  
tv->tv_sec = (long)(tmpres / 1000000UL);
tv->tv_usec = (long)(tmpres % 1000000UL);
}

if (NULL != tz)
{
if (!tzflag)
{
_tzset();
tzflag++;
}
tz->tz_minuteswest = _timezone / 60;
tz->tz_dsttime = _daylight;
}

return 0;
}


struct rusage {
struct timeval ru_utime; 
struct timeval ru_stime; 
long   ru_maxrss;        
long   ru_ixrss;         
long   ru_idrss;         
long   ru_isrss;         
long   ru_minflt;        
long   ru_majflt;        
long   ru_nswap;         
long   ru_inblock;       
long   ru_oublock;       
long   ru_msgsnd;        
long   ru_msgrcv;        
long   ru_nsignals;      
long   ru_nvcsw;         
long   ru_nivcsw;        
};



struct iovec
{
void*   iov_base;
size_t  iov_len;
};

#include<algorithm>
#elif defined(__GNUC__) && (defined(__linux__) || defined(__APPLE__))
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdlib.h>
inline static void posix_memalign_free(void* mem)
{
free(mem);
}

#else
#   error "unknown platform"
#endif

#endif 


