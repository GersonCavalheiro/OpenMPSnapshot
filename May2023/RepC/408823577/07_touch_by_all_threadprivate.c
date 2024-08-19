#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>
#include <omp.h>
#define N_default 10000
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
#ifdef DEBUG
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif
#define CPU_ID_ENTRY_IN_PROCSTAT 39
#define HOSTNAME_MAX_LENGTH      200
int read_proc__self_stat ( int, int * );
int get_cpu_id           ( void       );
double                 *array           = NULL;
unsigned long long int  mywork          = 0;
unsigned long long int  first_iteration = 0;
#pragma omp threadprivate(array, mywork, first_iteration)
int main( int argc, char **argv )
{
unsigned long long int N        = N_default;
int                    nthreads = 1;
struct  timespec ts;
if ( argc > 1 )
N = (unsigned long long int) atoll( *(argv+1) );
#if defined(_OPENMP)    
int endrun = 0;
#pragma omp parallel
{
nthreads = omp_get_num_threads();
#ifdef OUTPUT    
#pragma omp master
PRINTF("omp summation with %d threads\n", nthreads );
#endif
int me = omp_get_thread_num();
PRINTF("thread %2d is running on core %2d\n", me, get_cpu_id() );    
#pragma omp for schedule(static)
for ( unsigned long long int ii = 0; ii < N; ii++ )
{
first_iteration = ii;
ii = __UINTMAX_MAX__-1;
}
PRINTF("thread %d: first i is %d\n", me, first_iteration);
mywork = N/nthreads + 1;
if ( (array = (double*)calloc( mywork, sizeof(double) )) == NULL )
{
printf("I'm sorry, on thread %d there is not"
"enough memory to host %llu bytes\n",
me, 
mywork * sizeof(double) );
#pragma omp atomic
endrun += 1;
}
else
printf("thread %d: mywork is %Lu, array is at %p\n", me, mywork, array);
#pragma omp barrier
if ( !endrun )
{
#pragma omp for schedule(static)
for ( unsigned long long int ii = 0; ii < N; ii++ )
array[ ii - first_iteration ] = (double)ii;
}
}
if ( endrun )
{
printf("some problem in memory allocation\n");      
return endrun;
}
printf("out of the parallel regions the array location is %p\n", array);
PRINTF("Initialization done\n");
#else
if ( (array = (double*)calloc( N, sizeof(double) )) == NULL ) {
printf("I'm sorry, on some thread there is not"
"enough memory to host %lu bytes\n",
N * sizeof(double) ); return 1; }
for ( unsigned long long int ii = 0; ii < N; ii++ )
array[ii] = (double)ii;
#endif
long double S      = 0;                                   
double th_avg_time = 0;                                   
double th_min_time = 1e11;                                
double th_max_time = 0   ;                                
double tstart  = CPU_TIME;
#if !defined(_OPENMP)
for ( int ii = 0; ii < N; ii++ )                          
S += array[ii];                                         
#else
#pragma omp parallel reduction(+:th_avg_time)				reduction(min:th_min_time)						reduction(max:th_max_time)                               
{                                                         
struct  timespec myts;                                  
double mystart = CPU_TIME_th;                           
printf("Thread %d is accessing memory %p\n", omp_get_thread_num(), array );
#pragma omp for schedule(static) reduction(+:S)
for ( unsigned long long int ii = 0; ii < N; ii++ )
S += array[ii-first_iteration];
double mytime = CPU_TIME_th - mystart;
th_avg_time += mytime;
th_min_time  = mytime;
th_max_time  = mytime;
}
#endif
double tend = CPU_TIME;
printf("Sum is %Lg, process took %g of wall-clock time\n\n"
"<%g> sec of avg thread-time\n"
"<%g> sec of min thread-time\n"
"<%g> sec of max thread-time\n"       ,
S, tend - tstart, th_avg_time/nthreads, th_min_time, th_max_time );
return 0;
}
int get_cpu_id( void )
{
#if defined(_GNU_SOURCE)                              
return  sched_getcpu( );
#else
#ifdef SYS_getcpu                                     
int cpuid;
if ( syscall( SYS_getcpu, &cpuid, NULL, NULL ) == -1 )
return -1;
else
return cpuid;
#else      
unsigned val;
if ( read_proc__self_stat( CPU_ID_ENTRY_IN_PROCSTAT, &val ) == -1 )
return -1;
return (int)val;
#endif                                                
#endif
}
int read_proc__self_stat( int field, int *ret_val )
{
*ret_val = 0;
FILE *file = fopen( "/proc/self/stat", "r" );
if (file == NULL )
return -1;
char   *line = NULL;
int     ret;
size_t  len;
ret = getline( &line, &len, file );
fclose(file);
if( ret == -1 )
return -1;
char *savetoken = line;
char *token = strtok_r( line, " ", &savetoken);
--field;
do { token = strtok_r( NULL, " ", &savetoken); field--; } while( field );
*ret_val = atoi(token);
free(line);
return 0;
}
