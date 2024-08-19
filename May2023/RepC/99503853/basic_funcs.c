#include <stdio.h>
#include <omp.h>
int main()
{
printf("Maximum threads used for parallel region = %d\n",omp_get_max_threads());
#pragma omp parallel
{
printf("The default number of threads= %d\n",omp_get_num_threads());
printf("Thread %d is executing now\n",omp_get_thread_num() );
}
#pragma omp parallel num_threads(2)
{
printf("The number threads running=%d\n",omp_get_num_threads() );
}
omp_set_num_threads(3);
printf("Set number of threads to be used by default = %d\n",omp_get_num_threads());
}
