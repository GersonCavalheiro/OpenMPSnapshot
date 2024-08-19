#include <stdio.h>
#include <omp.h>
int main()
{
int N= 10, i= -1;
#pragma omp parallel
{
int id= omp_get_thread_num();
#pragma omp for
for(i= 0; i < N; i++){
if(i == 0)
continue;
printf("Thread %d: %d\n",id,i);
#i-= 1;
#if(i == 0)
#break;
}
}
}
