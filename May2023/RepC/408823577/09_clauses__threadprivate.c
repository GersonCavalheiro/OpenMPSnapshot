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
int  me, myN;
int *array;
#pragma omp threadprivate( me, myN, array )
#define DEFAULT 100000
int main( int argc, char **argv )
{
int    N = ( argc > 1 ? atoi(*(argv+1)) : DEFAULT);
#pragma omp parallel 
{
me = omp_get_thread_num();
int nthreads = omp_get_num_threads();
myN = (N / nthreads) + (me < N%nthreads);
array = (int*)calloc( myN, sizeof(int) );
printf("+ thread %d has got %d elements; local array "
"(address stored in %p) starts at %p\n",
me, myN, &array, array );
int max = ( myN > 3 ? 3 : myN );
for( int j = 0; j < max; j++ )
array[j] = me*1000 + j;
}
printf("\nnow we are again in a serial region\n\n");
#pragma omp parallel 
{
char buffer[200];
sprintf( buffer, "* thread %d :: ", me );
int max = ( myN > 3 ? 3 : myN );
for( int j = 0; j < max; j++ )
sprintf( &buffer[strlen(buffer)], "[%d] = %4d , ", j, array[j] );
printf("%s\n", buffer );
free(array);
}
return 0;
}
