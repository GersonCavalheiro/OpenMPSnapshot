#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define ITERATIONS 200000000
int main(void){
const int threads_num = 4;
int inc = 0;
double wtime = omp_get_wtime();
#pragma omp parallel num_threads(threads_num)
{
#pragma omp for
for(long i = 0;i < ITERATIONS; i++){	
#pragma omp atomic
inc++;
}	
int threadnum = omp_get_thread_num();
printf("Thread: %d\n", threadnum);
}
wtime = omp_get_wtime() - wtime;
printf("Increment is at %d after %d iterations.\n",inc, ITERATIONS);
printf("Time taken %f\n", wtime );
}
