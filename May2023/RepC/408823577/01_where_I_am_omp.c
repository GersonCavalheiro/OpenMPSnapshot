#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#if !defined(_OPENMP)
#error "OpenMP support is mandatory for this code"
#endif
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
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
char *proc_bind_names[] = { "false (no binding)",
"true",
"master",
"close",
"spread" };
nthreads = omp_get_num_threads();
int binding = omp_get_proc_bind();
printf("+ %d threads in execution - - proc bind is set to \"%s\"\n",
nthreads, proc_bind_names[binding] );
}
int me      = omp_get_thread_num();
int nplaces = omp_get_num_places();    
int place   = omp_get_place_num();
int nprocs  = omp_get_place_num_procs(place);
int proc_ids[nprocs];
omp_get_place_proc_ids( place, proc_ids );
int npplaces = omp_get_partition_num_places();
#pragma omp barrier
#pragma omp for ordered
for ( int i = 0; i < nthreads; i++)
#pragma omp ordered
{
printf("thread %2d: place %d, nplaces %d, nprocs %d, npplaces %d | procs here are: ",
me, place, nplaces, nprocs, npplaces );
for( int p = 0; p < nprocs; p++ )
printf("%d ", proc_ids[p]);
printf("\n");
}
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
