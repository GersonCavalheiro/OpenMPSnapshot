#include <stdio.h>
#include <omp.h>
#include <unistd.h>
int main()
{
int N= 10;
#pragma omp parallel
{
#pragma omp for nowait
for(int i= 0; i < N; i++){
printf("Thread %d got %d.\n",omp_get_thread_num(),i);
sleep(i);
printf("Thread %d finished %d.\n",omp_get_thread_num(),i);
}
printf("Thread %d reached here.\n",omp_get_thread_num());
}
}
