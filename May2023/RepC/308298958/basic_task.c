#include <omp.h>
#include <stdio.h>
#define N 10
int main(){
int i;
printf("Parallel For:\n");
#pragma omp parallel for
for (i = 0; i < N; i++){
printf("Thread %d runs iteration %d\n", omp_get_thread_num(), i);
}
printf("\nParallel For using Tasks:\n");
#pragma omp parallel
{
#pragma omp single
for (i = 0; i < N; i++){
#pragma omp task
printf("Thread %d runs iteration %d\n", omp_get_thread_num(), i);
}
}
return 0;
}
