#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#define N_default     1000  
int main( int argc, char **argv )
{
int N         = N_default;
int nthreads  = 1;
if ( argc > 1 )
{
N = atoi( *(argv+1) );
if ( argc > 2 )
nthreads = atoi( *(argv+2) );
}
if( nthreads > 1 )
omp_set_num_threads(nthreads);
#pragma omp parallel
{
int me       = omp_get_thread_num();
int nthreads = omp_get_num_threads();
int chunk    = N / nthreads;
int mod      = N % nthreads;
int my_first = chunk*me + ((me < mod)?me:mod);
int my_chunk = chunk + (mod > 0)*(me < mod);
#pragma omp single
printf("nthreads: %d, N: %d --- chunk is %d, reminder is %d\n", nthreads, N, chunk, mod);
printf("thread %d : from %d to %d\n", me, my_first, my_first+my_chunk);
} 
return 0;
}
