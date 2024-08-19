#include <omp.h>
main ()  {
int nthreads, tid;
omp_set_num_threads(10);
#pragma omp parallel private(nthreads, tid)
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
