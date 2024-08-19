#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void say_hello( void )
{
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
printf("Hello from thread %d of %d\n", my_rank, thread_count);
}
int main( int argc, char* argv[] )
{
int thr = 2;
if ( argc == 2 )
thr = atoi( argv[1] );
#pragma omp parallel num_threads(thr)
say_hello();
return 0;
}
