#include <stdio.h>
#include <omp.h>
#include <unistd.h>
int main()
{
int N= 8;
#pragma omp parallel for ordered
for(int i= 0; i < N; i++){
sleep(N-i);
printf("Thread %d got %d.\n",omp_get_thread_num(),i);
#pragma omp ordered
{
printf("Thread %d reached here.\n",omp_get_thread_num());
}
sleep(N-i);
printf("Thread %d outside ordered!!\n",omp_get_thread_num());
}
}
