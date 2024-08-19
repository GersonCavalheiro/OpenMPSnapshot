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
#include <omp.h>
#define N_default 1000000
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
typedef unsigned int uint;
double heavy_work_0( uint );
double heavy_work_1( uint );
double heavy_work_2( uint );
int main( int argc, char **argv )
{
int     N        = N_default;
int     nthreads = 1;
struct  timespec ts;
if ( argc > 1 )
N = atoi( *(argv+1) );
#if defined(_OPENMP)
#pragma omp parallel
#pragma omp single
nthreads = omp_get_num_threads();
printf("omp summation with %d threads\n", nthreads );
#else
printf("this code has not been compiled with OpenMP support\n");
#endif
double result = 0;
double tstart = CPU_TIME;
#if !defined(_OPENMP)
result = heavy_work_0(N) + heavy_work_1(N) + heavy_work_2(N);
#else
#pragma omp parallel shared(result)
{
#pragma omp sections reduction(+:result)
{
#pragma omp section
result += heavy_work_0(N);
#pragma omp section
result += heavy_work_1(N);
#pragma omp section
result += heavy_work_2(N);    
}
}
#endif
double tend = CPU_TIME;
printf("The result is %g\nrun took %g of wall-clock time\n\n",
result, tend - tstart );
return 0;
}
double heavy_work_0( uint N )
{
double guess = 3.141572 / 3;
for( int i = 0; i < N; i++ )
{
guess = exp( guess );
guess = sin( guess );
}
return guess;
}
double heavy_work_1( uint N )
{
double guess = 3.141572 / 3;
for( int i = 0; i < N; i++ )
{
guess = log( guess );
guess = exp( sqrt(guess)/guess );
}
return guess;
}
double heavy_work_2( uint N  )
{
double guess = 3.141572 / 3;
for( int i = 0; i < N; i++ )
{
guess = sqrt( guess );
guess = exp( 1+1.0/guess );
}
return guess;
}
