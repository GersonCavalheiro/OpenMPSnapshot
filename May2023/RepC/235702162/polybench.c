#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#include <math.h>
#ifdef _OPENMP
# include <omp.h>
#endif
#ifndef POLYBENCH_THREAD_MONITOR
# define POLYBENCH_THREAD_MONITOR 0
#endif
#ifndef POLYBENCH_CACHE_SIZE_KB
# define POLYBENCH_CACHE_SIZE_KB 32770
#endif
#include <omp.h> 
int polybench_papi_counters_threadid = 0;
double polybench_program_total_flops = 0;
#ifdef POLYBENCH_PAPI
# include <papi.h>
# define POLYBENCH_MAX_NB_PAPI_COUNTERS 96
#include "papi_counters.list"
#endif
double polybench_t_start;
double polybench_t_end;
unsigned long long polybench_c_start;
unsigned long long polybench_c_end;
static double rtclock()
{
#ifdef POLYBENCH_TIME
#else
return 0;
#endif
}
#ifdef POLYBENCH_CYCLE_ACCURATE_TIMER
#endif
void polybench_flush_cache()
{
int cs = ((32770 * 1024) / sizeof(double ));
double *flush = (double *)(calloc(cs,sizeof(double )));
int i;
double tmp = 0.0;
#pragma omp parallel for private (i) reduction (+:tmp) firstprivate (cs)
for (i = 0; i <= cs - 1; i += 1) {
tmp += flush[i];
}
(((void )(sizeof(((tmp <= 10.0?1 : 0))))) , ((
{
if (tmp <= 10.0) 
;
else 
__assert_fail("tmp <= 10.0","polybench.c",94,__PRETTY_FUNCTION__);
})));
free(flush);
}
#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
#endif
#ifdef POLYBENCH_PAPI
# ifndef POLYBENCH_NO_FLUSH_CACHE
# endif
#ifdef POLYBENCH_PAPI_VERBOSE
#endif
#endif
void polybench_prepare_instruments()
{
#ifndef POLYBENCH_NO_FLUSH_CACHE
polybench_flush_cache();
#endif
#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
#endif
}
void polybench_timer_start()
{
polybench_prepare_instruments();
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
polybench_t_start = rtclock();
#else
#endif
}
void polybench_timer_stop()
{
#ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
polybench_t_end = rtclock();
#else
#endif
#ifdef POLYBENCH_LINUX_FIFO_SCHEDULER
#endif
}
void polybench_timer_print()
{
#ifdef POLYBENCH_GFLOPS
#else
# ifndef POLYBENCH_CYCLE_ACCURATE_TIMER
printf("%0.6f\n",polybench_t_end - polybench_t_start);
# else
# endif
#endif
}
static void *xmalloc(size_t num)
{
void *nnew = (void *)0;
int ret = posix_memalign(&nnew,32,num);
if (!nnew || ret) {
fprintf(stderr,"[PolyBench] posix_memalign: cannot allocate memory");
exit(1);
}
return nnew;
}
void *polybench_alloc_data(unsigned long long n,int elt_size)
{
size_t val = n;
val *= elt_size;
void *ret = xmalloc(val);
return ret;
}
