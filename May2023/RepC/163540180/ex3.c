#include <omp.h>
#include <stdio.h>
int main()
{
int nthr;
printf("Please enter num threadsr: ");
scanf("%d", &nthr);
omp_set_num_threads(nthr);
#pragma omp parallel
{
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
}
