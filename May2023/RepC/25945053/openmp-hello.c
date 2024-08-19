#include <omp.h>
#include <stdio.h>
int main ()
{
int nthreads, tid;
omp_set_num_threads( 4 );
#pragma omp parallel private(tid)
{
tid = omp_get_thread_num();
printf("Hello World from thread = %d\n", tid);
if (tid == 0) 
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
}  
}
