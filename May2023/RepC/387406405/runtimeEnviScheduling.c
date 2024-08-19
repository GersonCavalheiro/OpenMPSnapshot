#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main()
{
int i,N=25;
#pragma omp parallel for num_threads(4)
for (i = 0; i < N; i++) 
{
printf("Thread %d is doing iteration %d.\n",
omp_get_thread_num( ), i);
}
return 0;
}
