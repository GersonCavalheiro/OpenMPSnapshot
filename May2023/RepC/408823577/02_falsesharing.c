#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <omp.h>
#define N_default 100
#define CPU_TIME_W (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec +	\
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_T (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +	\
(double)myts.tv_nsec * 1e-9)
#define CPU_TIME_P (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec +	\
(double)ts.tv_nsec * 1e-9)
#define CPU_ID_ENTRY_IN_PROCSTAT 39
#define HOSTNAME_MAX_LENGTH      200
int read_proc__self_stat ( int, int * );
int get_cpu_id           ( void       );
int main( int argc, char **argv )
{
int     N        = N_default;
int     nthreads = 1;
struct  timespec ts;
double *array;
if ( argc > 1 )
N = atoi( *(argv+1) );
if ( (array = (double*)calloc( N, sizeof(double) )) == NULL )
{
printf("I'm sorry, there is not enough memory to host %lu bytes\n", N * sizeof(double) );
return 1;
}
#pragma omp parallel
{
#pragma omp master
{
nthreads = omp_get_num_threads();
printf("omp summation with %d threads\n", nthreads );
}
int me = omp_get_thread_num();
#pragma omp critical
printf("thread %2d is running on core %2d\n", me, get_cpu_id() );
}
for ( int ii = 0; ii < N; ii++ )
array[ii] = (double)ii;
double S[ nthreads ];                                     
double th_avg_time = 0;                                   
double th_min_time = (1<<30);                             
double tstart  = CPU_TIME_W;  
memset( S, 0, nthreads*sizeof(double) );
#pragma omp parallel shared(S)
{    
int    me       = omp_get_thread_num();
double mytstart = CPU_TIME_T; 
#pragma omp for
for ( int ii = 0; ii < N; ii++ )
S[me] += array[ii];
double myt      = CPU_TIME_T - mytstart;;
#pragma omp atomic update
th_avg_time += myt;
#pragma omp atomic update
th_min_time  = (th_min_time > myt? myt : th_min_time);
}
if ( nthreads > 1 )                                       
for ( int ii = 1; ii < nthreads; ii++ )                 
S[0] += S[ii];                                        
double tend = CPU_TIME_W;
printf("Sum is %g, process took %g of wall-clock time\n\n"
"<%g> sec of avg thread-time\n"
"<%g> sec of min thread-time\n",
S[0], tend - tstart, th_avg_time/nthreads, th_min_time );
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
