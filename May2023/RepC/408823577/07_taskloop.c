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
#define min_default    100   
#define max_default    20000 
#define NANO_PAUSE    100   
#define uSEC          1000  
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
#if defined(DEBUG)
#define PRINTF(...) printf(__VA_ARGS__);
#define PRINTFS(...) _Pragma("omp single")	\
printf(__VA_ARGS__);
#else
#define PRINTF(...)
#define PRINTFS(...)
#endif
typedef unsigned int uint;
double heavy_work_0( uint );
double heavy_work_1( uint );
double heavy_work_2( uint );
int main( int argc, char **argv )
{
int N         = N_default;
int min_value = min_default;
int max_value = max_default;
struct  timespec ts;
if ( argc > 1 )
{
N = atoi( *(argv+1) );
if( argc > 2 )
max_value = atoi( *(argv+2) );
}
srand48(1234321);
double result = 0;
int *array = (int*)malloc( N*sizeof(double) );
#if !defined(_OPENMP)
printf("serial summation\n" );
double tstart = CPU_TIME;
for( int ii = 0; ii < N; ii++ )    
array[ii] = min_value + lrand48() % max_value;     
for( int ii = 0; ii < N; ii++ )
result += heavy_work_0(array[ii]) +
heavy_work_1(array[ii]) + heavy_work_2(array[ii]);
double tend = CPU_TIME;
#else
double tstart = CPU_TIME;
#pragma omp parallel proc_bind(close) 
{
#pragma omp single nowait
{
#pragma omp taskloop num_tasks(N/10) reduction(+:result)
for( int ii = 0; ii < N; ii++ )
{
array[ii] = min_value + lrand48() % max_value;
result += heavy_work_0(array[ii]) +
heavy_work_1(array[ii]) +
heavy_work_2(array[ii]);
}
}
PRINTF("* initializer thread: initialization lasted %g seconds\n", CPU_TIME_th - tstart );
} 
double tend = CPU_TIME;
#endif
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
