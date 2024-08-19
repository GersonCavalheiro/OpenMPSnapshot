#if defined (__vxworks)
#ifdef __RTP__
#include <time.h>
#include <version.h>
#if (_WRS_VXWORKS_MAJOR == 7) || (_WRS_VXWORKS_MINOR != 0)
#include <sys/time.h>
#endif
#else
#include <sys/times.h>
#endif
#elif defined (__nucleus__)
#include <time.h>
#else
#include <sys/time.h>
#endif
#ifdef __MINGW32__
#include "mingw32.h"
#if STD_MINGW
#include <winsock.h>
#endif
#endif
void
__gnat_timeval_to_duration (struct timeval *t, long long *sec, long *usec)
{
*sec  = (long long) t->tv_sec;
*usec = (long) t->tv_usec;
}
void
__gnat_duration_to_timeval (long long sec, long usec, struct timeval *t)
{
t->tv_sec = sec;
t->tv_usec = usec;
}
