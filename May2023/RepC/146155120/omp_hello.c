#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main (int argc, char *argv[])
{
int nthreads, tid;
clock_t begin = clock();
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
clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
printf("Elapsed CPU time is %f\n",time_spent);
}