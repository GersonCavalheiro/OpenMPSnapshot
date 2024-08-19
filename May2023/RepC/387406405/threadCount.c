#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define B 1000000000.0
void hello(void);
int count=0;
int main(intargc,char* argv)
{
int thread_count = 10000;
struct timespec start,end;
clock_gettime(CLOCK_REALTIME,&start);
#pragma omp parallel num_threads(thread_count)
hello();
printf("total threads = %d\n",count);
clock_gettime(CLOCK_REALTIME,&end);
double time=(end.tv_sec-start.tv_sec)+(end.tv_nsec-start.tv_nsec)/B;
printf("time taken is%f\n",time);
return 0;
}
void hello(void)
{
int my_rank=omp_get_thread_num();
int thread_count=omp_get_num_threads();
count++;
}
