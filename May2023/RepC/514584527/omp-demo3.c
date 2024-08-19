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
omp_set_num_threads(4);
#pragma omp parallel
say_hello();
return 0;
}
