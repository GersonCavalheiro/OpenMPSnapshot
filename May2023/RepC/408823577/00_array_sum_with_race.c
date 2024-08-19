#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#define N_default 100
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec +	\
(double)ts.tv_nsec * 1e-9)
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
#ifndef _OPENMP
printf("serial summation\n");
#else
#pragma omp parallel
{
#pragma omp master
{
nthreads = omp_get_num_threads();
printf("omp summation with %d threads\n", nthreads );
}
}
#endif
for ( int ii = 0; ii < N; ii++ )                          
array[ii] = (double)ii;                                 
double S       = 0;                                       
double tstart  = CPU_TIME;
#if !defined(_OPENMP)
for ( int ii = 0; ii < N; ii++ )                          
S += array[ii];                                         
#else
#pragma omp parallel for 
for ( int ii = 0; ii < N; ii++ )
S += array[ii];
#endif
double tend = CPU_TIME;
printf("Sum is %g, process took %g of wall-clock time\n", S, tend - tstart );
free( array );
return 0;
}
