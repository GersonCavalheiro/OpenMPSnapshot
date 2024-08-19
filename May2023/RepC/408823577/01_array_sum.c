#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#define N_default 100
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec + \
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
int main( int argc, char **argv )
{
int     N        = N_default;
int     nthreads = 1;
struct  timespec ts;
struct  timespec myts;
double *array;
if ( argc > 1 )
N = atoi( *(argv+1) );
if ( (array = (double*)malloc( N * sizeof(double) )) == NULL )
{
printf("I'm sorry, there is not enough memory to host %lu bytes\n", N * sizeof(double) );
return 1;
}
#ifndef _OPENMP
printf("serial summation\n");
#else
#pragma omp parallel
#pragma omp master
nthreads = omp_get_num_threads();
printf("omp summation with %d threads\n", nthreads );
#endif
double _tstart = CPU_TIME_th;
for ( int ii = 0; ii < N; ii++ )
array[ii] = (double)ii;                                 
double _tend = CPU_TIME_th;
printf("init time is %g\n", _tend - _tstart);
double S           = 0;                                   
double th_avg_time = 0;                                   
double th_min_time = 1e11;                                
double tstart  = CPU_TIME;
#if !defined(_OPENMP)
for ( int ii = 0; ii < N; ii++ )                          
S += array[ii];                                         
#else
#pragma omp parallel reduction(+:th_avg_time) reduction(min:th_min_time)                                
{                                                         
struct  timespec myts;                                  
double mystart = CPU_TIME_th;                           
#pragma omp for reduction(+:S)                              
for ( int ii = 0; ii < N; ii++ )
S += array[ii];
th_avg_time   = CPU_TIME_th - mystart; 
th_min_time   = CPU_TIME_th - mystart;     
}
#endif
double tend = CPU_TIME;                                   
printf("Sum is %g, process took %g of wall-clock time\n\n"
"<%g> sec of avg thread-time\n"
"<%g> sec of min thread-time\n",
S, tend - tstart, th_avg_time/nthreads, th_min_time );
free( array );
return 0;
}
