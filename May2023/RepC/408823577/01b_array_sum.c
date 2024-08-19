#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#if !defined(_OPENMP)
#error "OpenMP support needed for this code"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#define N_default 100
int main( int argc, char **argv )
{
int     N        = N_default;
int     nthreads = 1;  
double *array;
#pragma omp parallel
#pragma omp master
nthreads = omp_get_num_threads();
printf("omp summation with %d threads\n", nthreads );
if ( (array = (double*)calloc( N, sizeof(double) )) == NULL )
{
printf("I'm sorry, there is not enough memory to host %lu bytes\n", N * sizeof(double) );
return 1;
}
for ( int ii = 0; ii < N; ii++ )
array[ii] = (double)ii;
double S           = 0;                                   
#pragma omp parallel 
{
int me      = omp_get_thread_num();
int i, first = 1;
printf("thread %d : &i is %p\n", me, &i);
#pragma omp for reduction(+:S)                              
for ( i = 0; i < N; i++ )
{
if ( first ) {
printf("\tthread %d : &loopcounter is %p\n", me, &i); first = 0;}
S += array[i];
}    
}
printf("Sum is %g\n\n", S );
free( array );
return 0;
}
