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
#define N_default 1000
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
#ifdef OUTPUT
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif
#define CPU_ID_ENTRY_IN_PROCSTAT 39
#define HOSTNAME_MAX_LENGTH      200
int read_proc__self_stat ( int, int * );
int get_cpu_id           ( void       );
int main( int argc, char **argv )
{
int     N        = N_default;
int     nthreads = 1;
struct  timespec ts, myts;
double *array;
if ( argc > 1 )
N = atoi( *(argv+1) );
if ( (array = (double*)malloc( N * sizeof(double) )) == NULL ) {
printf("I'm sorry, on some thread there is not"
"enough memory to host %lu bytes\n",
N * sizeof(double) ); return 1; }
#if defined(_OPENMP)  
#pragma omp parallel
{
#pragma omp master
{
nthreads = omp_get_num_threads();
PRINTF("omp summation with %d threads\n", nthreads );
}
int me = omp_get_thread_num();
#pragma omp critical
PRINTF("thread %2d is running on core %2d\n", me, get_cpu_id() );    
}
#endif
double _tstart = CPU_TIME_th;
#pragma omp parallel for
for ( int ii = 0; ii < N; ii++ )
array[ii] = (double)ii;
double _tend = CPU_TIME_th;
printf("init takes %g\n", _tend - _tstart);
double S           = 0;                                   
double th_avg_time = 0;                                   
double th_min_time = 1e11;                                
double tstart  = CPU_TIME;
#if !defined(_OPENMP)
for ( int ii = 0; ii < N; ii++ )                          
S += array[ii];                                         
#else
#pragma omp parallel reduction(+:th_avg_time)				reduction(min:th_min_time)                                
{                                                         
struct  timespec myts;                                  
double mystart = CPU_TIME_th;                           
#pragma omp for reduction(+:S)                              
for ( int ii = 0; ii < N; ii++ )
S += array[ii];
th_avg_time  = CPU_TIME_th - mystart; 
th_min_time  = CPU_TIME_th - mystart; 
}
#endif
double tend = CPU_TIME;
printf("Sum is %g, process took %g of wall-clock time\n\n",
S, tend - tstart);
#if defined(_OPENMP)
printf("<%g> sec of avg thread-time\n"
"<%g> sec of min thread-time\n",
th_avg_time/nthreads, th_min_time );
#endif
free( array );
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
