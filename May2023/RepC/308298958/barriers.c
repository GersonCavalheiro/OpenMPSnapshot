#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(){
#pragma omp parallel
{
printf("Thread %d running inside of the parallel region\n", omp_get_thread_num());
}
printf("Main thread after implicit barrier!\n");
#pragma omp parallel
{
printf("Thread %d running before explicit barrier\n", omp_get_thread_num());
printf("Thread %d running after explicit barrier\n", omp_get_thread_num());
}
printf("Main thread after implicit barrier!\n");
#pragma omp parallel
{
printf("Thread %d running before explicit barrier\n", omp_get_thread_num());
#pragma omp barrier
printf("Thread %d running after explicit barrier\n", omp_get_thread_num());
}
printf("Main thread after implicit and explicit barriers!\n");
return 0;
}
