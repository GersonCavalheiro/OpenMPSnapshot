#include <stdio.h>
#include <omp.h>
int main()
{
int threads= -1, id= -1, cores= -1;
omp_set_num_threads(2);
#pragma omp parallel num_threads(9)
{
threads= omp_get_num_threads();
id= omp_get_thread_num();
cores= omp_get_num_procs();
#pragma omp master
{
printf("Total cores= %d\n",cores);
printf("Total threads= %d\n",threads);
printf("Thread id= %d\n",id);
printf("Thread limit= %d\n",omp_get_thread_limit());
}
}
}
