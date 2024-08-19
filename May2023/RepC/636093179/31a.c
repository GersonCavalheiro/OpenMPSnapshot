#include <stdio.h>
#include <omp.h>
#include <unistd.h>
int main()
{
int N= 10;
#pragma omp parallel
{
printf("Thread %d.\n",omp_get_thread_num());
sleep(omp_get_thread_num());
printf("Thread %d finished.\n",omp_get_thread_num());
}
printf("Thread %d reached here.\n",omp_get_thread_num());
}
