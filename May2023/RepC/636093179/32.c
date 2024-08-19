#include <stdio.h>
#include <omp.h>
#include <unistd.h>
int main()
{
int N= 4;
#pragma omp parallel
{
#pragma omp for schedule(dynamic) nowait
for(int i= 0; i < N; i++){
printf("Thread %d in first loop got %d.\n",omp_get_thread_num(),i);
sleep(i);
}
printf("Thread %d reached after first for loop.\n",omp_get_thread_num());
#pragma omp for schedule(static) nowait
for(int i= 0; i < N; i++){
printf("Thread %d in second loop got %d.\n",omp_get_thread_num(),i);
sleep(i);
}
printf("Thread %d reached after second for loop.\n",omp_get_thread_num());
}
printf("Outside Parallel region\n");
}
