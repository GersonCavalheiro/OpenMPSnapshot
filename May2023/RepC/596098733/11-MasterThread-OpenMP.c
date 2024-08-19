#include <stdio.h>
#include <omp.h>
int main()
{
int nthreads, tid;
#pragma omp parallel private(tid)
{
tid = omp_get_thread_num();
#pragma omp master
{
nthreads = omp_get_num_threads();
printf("I am the master, pID %d. Number of threads = %d\n", tid, nthreads);
}
printf("Hello World from thread %d\n", tid);
}
return 0;
}
