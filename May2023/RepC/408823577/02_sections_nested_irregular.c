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
#define chunkf_default 10    
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
#if !defined(NTHREADS)    
#define NTHREADS 3
#endif
#define SEED 1234321
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
int chunkf    = chunkf_default;
int nthreads  = 1;
int chunk     = N / chunkf_default;
struct  timespec ts;
if ( argc > 1 )
{
N = atoi( *(argv+1) );
if( argc > 2 )
{
max_value = atoi( *(argv+2) );
if( argc > 3 )
chunkf = atoi( *(argv+3) );
}
}
double result = 0;
int *array = (int*)malloc( N*sizeof(double) );
srand48(SEED);
#if !defined(_OPENMP)
printf("serial summation\n" ); 
double tstart = CPU_TIME;
#if !defined (MIMIC_SLOWER_INITIALIZATION)
for( int ii = 0; ii < N; ii++ )
array[ii] = min_value + lrand48() % max_value;
#else
struct timespec nanot = {0.1, 0};
nanosleep( &nanot, NULL );
int first = 0;
int last  = chunk;
int idx   = 0;
while( first < N )
{
last = (last >= N)?N:last;
for( int kk = first; kk < last; kk++, idx++ )
array[idx] = min_value + lrand48() % max_value;
first += chunk;
last  += chunk;
nanot.tv_sec = 0.01 + lrand48() % 2;
nanosleep( &nanot, NULL );
}
#endif
#if defined(DEBUG)  
double partial_result1, partial_result2, partial_result3 = 0;
for( int ii = 0; ii < N; ii++ )
partial_result1 += heavy_work_0(array[ii]);
for( int ii = 0; ii < N; ii++ )
partial_result2 += heavy_work_1(array[ii]);
for( int ii = 0; ii < N; ii++ )
partial_result3 += heavy_work_2(array[ii]);
result = partial_result1 + partial_result2 + partial_result3;
double tend = CPU_TIME;
printf("partial results are: %g %g %g\n", partial_result1, partial_result2, partial_result3 );    
#else
for( int ii = 0; ii < N; ii++ )
result += heavy_work_0(array[ii]) +
heavy_work_1(array[ii]) + heavy_work_2(array[ii]);
double tend = CPU_TIME;
#endif
#else
double tstart = CPU_TIME;
omp_set_num_threads( NTHREADS+1 );
#pragma omp parallel
#pragma omp single
nthreads= omp_get_num_threads();
PRINTF("omp summation with %d threads\n", nthreads );
if ( nthreads < NTHREADS+1 )
{
printf("something odd happened: you did not get"
" as many threads as requested (%d instead of %d)\n",
nthreads, NTHREADS+1 );
}
int    semaphore = 0;
#pragma omp parallel proc_bind(spread) shared(result) firstprivate(nthreads)
{
#pragma omp single
{
if( ! omp_get_nested() )
printf(">>> WARNING: nesting is not active, so you'll loose a lot of speedup");
}
#pragma omp sections reduction(+:result) firstprivate( chunk )
{
#pragma omp section 
{	
int idx   = 0;
int first = 0;
int last  = chunk;
#if defined (MIMIC_SLOWER_INITIALIZATION)
struct timespec nanot = {0.1, 0};
nanosleep( &nanot, NULL );
#endif
#if defined(DEBUG)
struct timespec myts;
double tstart = CPU_TIME_th;
int    me     = omp_get_thread_num();
#endif
while( first < N )
{
last = (last >= N)? N : last;
for( int kk = first; kk < last; kk++, idx++ )
array[idx] = min_value + lrand48() % max_value;
#pragma omp atomic write
semaphore = last;
PRINTF("* initializer (thread %d) : %g sec, initialized chunk from %d to %d\n",
me, CPU_TIME_th - tstart, first, last);
first += chunk;
last  += chunk;
#if defined (MIMIC_SLOWER_INITIALIZATION)
{
nanot.tv_sec = 0.01 + lrand48() % 2;
nanosleep( &nanot, NULL );
}
#endif
}
PRINTF("* initializer thread: initialization lasted %g seconds\n", CPU_TIME_th - tstart );
}
#pragma omp section 
{
int    mysemaphore = 0;
int    last        = 0;
double myresult    = 0;
const struct timespec nanot = {0, NANO_PAUSE};
#pragma omp parallel reduction(+:myresult) shared(last, mysemaphore) proc_bind(close)
{
int inner_nthreads = omp_get_num_threads();
int me             = omp_get_thread_num();
PRINTFS("- \t section 1 :: %d threads active\n", inner_nthreads);
while( last < N )
{
#pragma omp single
{
#pragma omp atomic read
mysemaphore = semaphore;
while( mysemaphore == last )
{
nanosleep(&nanot, NULL);
#pragma omp atomic read
mysemaphore = semaphore;
}
}
int my_chunk = (mysemaphore-last)/inner_nthreads;
int my_last  = my_chunk*(me+1);
int my_first = my_chunk*me;
my_last  = (my_last > N)? N : my_last;		  
#pragma omp single nowait
{
PRINTF("- \t\t section 1 :: processing from %d to %d\n", last, mysemaphore);
last = mysemaphore;	     
}
for( int ii = my_first; ii < my_last; ii++)
myresult += heavy_work_0(array[ii]);	      
}
}
PRINTF("- \t section 1 :: result is %g\n", myresult);
result += myresult;
}  
#pragma omp section 
{
int    mysemaphore = 0;
int    last        = 0;
double myresult    = 0;
const struct timespec nanot = {0, NANO_PAUSE};
#pragma omp parallel reduction(+:myresult) shared(last) proc_bind(close)
{
int inner_nthreads = omp_get_num_threads();
int me             = omp_get_thread_num();
PRINTFS("- \t section 2 :: %d threads active\n", inner_nthreads);
while( last < N )
{
#pragma omp single
{
#pragma omp atomic read
mysemaphore = semaphore;
while( mysemaphore == last )
{
nanosleep(&nanot, NULL);
#pragma omp atomic read
mysemaphore = semaphore;
}
}
int my_chunk = (mysemaphore-last)/inner_nthreads;
int my_last  = my_chunk*(me+1);
int my_first = my_chunk*me;
my_last  = (my_last > N)? N : my_last;		  
#pragma omp single nowait
{      	
PRINTF("- \t\t section 2 :: processing from %d to %d\n", last, mysemaphore);	      
last = mysemaphore;
}
for( int ii = my_first; ii < my_last; ii++)
myresult += heavy_work_1(array[ii]);	      
}
}
PRINTF("- \t section 2 :: result is %g\n", myresult);
result += myresult;
}  
#pragma omp section 
{
int    mysemaphore = 0;
int    last        = 0;
double myresult    = 0;
const struct timespec nanot = {0, NANO_PAUSE};
#pragma omp parallel reduction(+:myresult) shared(last) proc_bind(close)
{
int inner_nthreads = omp_get_num_threads();
int me             = omp_get_thread_num();
PRINTFS("-\t section 3 :: %d threads active\n", inner_nthreads);
while( last < N )
{
#pragma omp single
{
#pragma omp atomic read
mysemaphore = semaphore;
while( mysemaphore == last )
{
nanosleep(&nanot, NULL);
#pragma omp atomic read
mysemaphore = semaphore;
}
}
int my_chunk = (mysemaphore-last)/inner_nthreads;
int my_last  = my_chunk*(me+1);
int my_first = my_chunk*me;
my_last  = (my_last > N)? N : my_last;		  
#pragma omp single nowait
{
PRINTF("- \t\t section 3 :: processing from %d to %d\n", last, mysemaphore);	      
last = mysemaphore;
}
for( int ii = my_first; ii < my_last; ii++)
myresult += heavy_work_2(array[ii]);	      
}
}
PRINTF("- \t section 3 :: result is %g\n", myresult);
result += myresult;
}  
} 
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
