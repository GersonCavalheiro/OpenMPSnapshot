#include <stdio.h>
#include <omp.h>
#include <unistd.h>
int main()
{
int N= 10, i= -1;
#pragma omp parallel
{
int id= omp_get_thread_num();
#pragma omp for schedule(guided,2)
for(i= 0; i < N; i++){
printf("Thread %d got %d\n",id,i);
sleep(2);
}
}
}
