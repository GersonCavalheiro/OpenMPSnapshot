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
#define N_default      20000   
#define max_default    20000   
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
#if !defined(NTHREADS)    
#define NTHREADS 3
#endif
#if defined(DEBUG)
#define PRINTF(...) printf(__VA_ARGS__);
#else
#define PRINTF(...)
#endif
typedef unsigned int uint;
double heavy_work_0( uint );
double heavy_work_1( uint );
double heavy_work_2( uint );
int main( int argc, char **argv )
{
int      N         = N_default;
int      max_value = max_default;
int      nthreads  = 1;
struct  timespec ts, myts;
if ( argc > 1 )
{
N = atoi( *(argv+1) );
if( argc > 2 )
max_value = atoi( *(argv+2) );
}
srand48(1234321);
double result = 0;
int *array = (int*)malloc( N*sizeof(double) );
for( int ii = 0; ii < N; ii++ )
array[ii] = 100 + (lrand48() % max_value);
#if !defined(_OPENMP)
printf("serial summation\n" );
double tstart = CPU_TIME;
for( int ii = 0; ii < N; ii++ )
result += heavy_work_0(array[ii]) + heavy_work_1(array[ii]) + heavy_work_2(array[ii]);
#else
printf("omp summation\n" );
omp_set_num_threads( NTHREADS+1 );
#pragma omp parallel
#pragma omp single
nthreads= omp_get_num_threads();
PRINTF("omp summation with %d threads\n", nthreads );
if ( nthreads < NTHREADS )
{
printf("something odd happened: you did not get"
" as many threads as requested (%d instead of %d)\n",
nthreads, NTHREADS );
}
double tstart = CPU_TIME;
#pragma omp parallel proc_bind(spread) shared(result) firstprivate(N)
{
#pragma omp sections reduction(+:result)
{
#pragma omp section
{
double myresult = 0;
#pragma omp parallel for reduction(+:myresult)
for( int jj = 0; jj < N; jj++ )
myresult += heavy_work_0( array[jj] );
result += myresult;	    
}
#pragma omp section
{
double myresult = 0;
#pragma omp parallel for reduction(+:myresult)
for( int jj = 0; jj < N; jj++ )
myresult += heavy_work_1( array[jj] );
result += myresult;	    
}
#pragma omp section
{
double myresult = 0;
#pragma omp parallel for reduction(+:myresult)
for( int jj = 0; jj < N; jj++ )
myresult += heavy_work_2( array[jj] );
result += myresult;	    
}
}
}
#endif
double tend = CPU_TIME;
free(array);
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
