#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#define DEFAULT 10
int main( int argc, char **argv )
{
int  i = (argc > 1 ? atoi(*(argv+1)) : DEFAULT );
int  nthreads;
int *array;
#pragma omp parallel
#pragma omp master
nthreads = omp_get_num_threads();
array = (int*)calloc( nthreads, sizeof(int) );
#pragma omp parallel firstprivate( i, array )
{
int me = omp_get_thread_num();
array[me] = i + me;   
array = NULL;         
}
for( int j = 0; j < nthreads; j++ )
printf("entry %3d is %3d (expected was %3d)\n",
j, array[j], i + j );
free(array);
return 0;
}
