#include<stdio.h>
#include<omp.h>
int main()
{
#pragma omp parallel num_threads(4)
{
printf("Inside parallel block. I am %d\n", omp_get_thread_num());
}
return 0;
}
