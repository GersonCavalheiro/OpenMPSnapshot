#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
#include<unistd.h>
int main()
{
int i;
#pragma omp parallel
{
#pragma omp for schedule(static)
for(i=0;i<8;i++)
{
#pragma omp
sleep(i);
printf("Thread=%d completed iteration with time %d seconds\n",omp_get_thread_num(),i);
}
}
sleep(5);
printf("-------------");
#pragma omp parallel
{
#pragma omp for schedule(dynamic)
for(i=0;i<8;i++)
{
sleep(i);
printf("Thread=%d completed iteration with time %d seconds\n",omp_get_thread_num(),i);
}
}
}
