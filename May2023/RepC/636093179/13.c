#include <stdio.h>
#include <omp.h>
void fun()
{
printf("Entering fun...\n");
#pragma omp parallel
{
printf("I am %d\n",omp_get_thread_num());
}
printf("Leaving fun...\n");
}
int main()
{
omp_set_num_threads(2);
omp_set_nested(1);
omp_set_max_active_levels(2);
printf("Max active levels= %d\n",omp_get_max_active_levels());
#pragma omp parallel
{
int threads= -1, id= -1, cores= -1;
threads= omp_get_num_threads();
id= omp_get_thread_num();
printf("Total threads= %d\n",threads);
printf("Thread id= %d\n",id);
fun();
}
}
