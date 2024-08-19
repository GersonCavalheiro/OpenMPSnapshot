#include <stdio.h>
#include <omp.h>
int main()
{
int i;
#pragma omp parallel
{
#pragma omp for
for(i=0;i<4;i++)
{
printf("Executing thread %d\n",omp_get_thread_num());
}
}
printf("----Using ordered----\n");
#pragma omp parallel
{
#pragma omp for ordered
for(i=0;i<4;i++)
{
#pragma omp ordered
printf("Executing thread %d\n",omp_get_thread_num());
}
}
}
