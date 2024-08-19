#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mypapi.h"		                   
#ifndef PIPELINE
#define PIPELINE 0
#endif
#define XSTR(x) #x
#define STR(x) XSTR(x)
#define BASENAME v
#define SUFFIX   .c
#define ALIGN 32
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec +	\
(double)ts.tv_nsec * 1e-9)
#ifndef _MEM_CLOCK        
#define _MEM_CLOCK 2133	  
#endif
#ifndef _MEM_WIDTH        
#define _MEM_WIDTH 64	  
#endif
#ifndef _MEM_CHNS         
#define _MEM_CHNS 2
#endif
int main(int argc, char **argv)
{
double *array1, *array2;                         
int     N;                                       
struct timespec ts;                              
if(argc > 1)
N = atoi(*(argv+1));
else
N = 1000000;				   
array1 = (double*)aligned_alloc(ALIGN, 2*N * sizeof(double));
array2 = array1 + N;
srand48(12983476);				   
printf("generating %d numbers..", 2*N); fflush(stdout);
{
int N_ = 2*N;
for ( int i = 0; i < N_; i++ )
array1[i] = drand48();
}
printf ( "done\n" );
PAPI_INIT;
#define ITER 10
double std_dev =0,  avg_timing = 0, min_time = N;
double sum = 0;
double tstart_all = CPU_TIME;
for(int k = 0; k < ITER; k++)
{
sum = 0;
double tstart = CPU_TIME;
PAPI_START_CNTR;
#include STR(BASENAME PIPELINE SUFFIX)
PAPI_STOP_CNTR;
double timing = CPU_TIME - tstart;
avg_timing += timing;
std_dev  += timing * timing;
if ( timing < min_time )
min_time = timing;
}
double tend_all = CPU_TIME - tstart_all;
avg_timing /= ITER;
std_dev = sqrt(std_dev / ITER - avg_timing*avg_timing);
printf( "sum is %g\ntime is :%g (min %g, std_dev %g, all %g)\n",
sum, avg_timing, min_time, std_dev, tend_all );
double max_GB_per_sec          = (double)_MEM_CLOCK / 1000 * _MEM_CHNS * (_MEM_WIDTH / 8);
double transfer_rate_in_GB_sec = (double)N*2*sizeof(double) / (1024*1024*1024) / avg_timing;  
printf("transfer rate was %6.3g GB/sec (%7.4g%% of theoretical max that is %5.2g GB/sec)\n",
transfer_rate_in_GB_sec, transfer_rate_in_GB_sec / max_GB_per_sec * 100, max_GB_per_sec);
#ifdef USE_PAPI
printf( "\tIPC: %4.2g\n"
"\t[ time-per-element: %6.4gsec  -  cycles-per-element: %9.6g ]\n",
(double)papi_values[0] / papi_values[1],
avg_timing / (ITER*N),
(double)papi_values[1] / (ITER*N) );
#endif
memset ( array1, 0, sizeof(double)*2*N );
free(array1);
return 0;
}
