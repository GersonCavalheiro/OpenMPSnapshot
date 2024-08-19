#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#if !defined(_OPENMP)
#error "OpenMP support required for this code"
#endif
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>
#include <omp.h>
#define N_default 1000
#define CPU_ID_ENTRY_IN_PROCSTAT 39
#define HOSTNAME_MAX_LENGTH      200
int read_proc__self_stat ( int, int * );
int get_cpu_id           ( void       );
int main( int argc, char **argv )
{
int nthreads           = 1;
int nthreads_requested = 1;
if ( argc > 1 )
nthreads_requested = atoi( *(argv+1) );
if ( nthreads_requested > 1 )
omp_set_num_threads( nthreads_requested ); 
char *places = getenv("OMP_PLACES");
char *bind   = getenv("OMP_PROC_BIND");
if ( places != NULL )
printf("OMP_PLACES is set to %s\n", places);
if ( bind != NULL )
printf("OMP_PROC_BINDING is set to %s\n", bind);
#pragma omp parallel
{
#pragma omp master
{
nthreads = omp_get_num_threads();
printf("+ %d threads in execution - -\n", nthreads );
}
int me = omp_get_thread_num();
#pragma omp critical
printf("thread %2d is running on core %2d\n", me, get_cpu_id() );
#pragma omp barrier
#ifdef SPY
#define REPETITIONS 10000
#define ALOT        10000000000
long double S = 0;
for( int j = 0; j < REPETITIONS; j++ )
#pragma omp for
for( unsigned long long i = 0; i < ALOT; i++ )
S += (long double)i;
#endif
}
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
