#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
int nthreads, i, tid; 
float total = 0.0;
#pragma omp parallel private (i, tid)
{
#pragma omp single
{
nthreads = omp_get_num_threads();
printf("Number of threads = %d\n", nthreads);
}
tid = omp_get_thread_num();
printf("Thread %d is starting...\n", tid);
#pragma omp for reduction(+:total)
for (i = 0; i < 100; i++)
total += i*1.0;
printf ("Thread %d is done! Total= %f\n", tid, total);
}
}