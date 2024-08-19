#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <omp.h>
#define THREADS 4
#define N 16
int main ()
{
int i;
double start_time;
start_time = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(THREADS)
for (i = 0; i < N; i++) {
sleep(i);
printf("Thread %d has completed iteration %d.\n", omp_get_thread_num( ), i);
}
printf("All done!\n");
printf("Execution_time: %f s\n",omp_get_wtime()-start_time);
return 0;
}
