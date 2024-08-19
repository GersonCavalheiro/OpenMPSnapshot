#include <stdio.h>
#include <omp.h>
#define MAX_THREADS 4
static long num_steps = 100000000;
double step;
int main ()
{
int i,j;
double pi, full_sum = 0.0;
double start_time, run_time;
double sum[MAX_THREADS];
step = 1.0/(double) num_steps;
for(j=1;j<=MAX_THREADS ;j++){
omp_set_num_threads(j);
full_sum = 0.0;
start_time = omp_get_wtime();
#pragma omp parallel private(i)
{
int id = omp_get_thread_num();
int numthreads = omp_get_num_threads();
double x;
double partial_sum = 0;
#pragma omp single
printf(" num_threads = %d",numthreads);
for (i=id;i< num_steps; i+=numthreads){
x = (i+0.5)*step;
partial_sum += + 4.0/(1.0+x*x);
}
#pragma omp critical
full_sum += partial_sum;
}
pi = step * full_sum;
run_time = omp_get_wtime() - start_time;
printf("\n pi is %f in %f seconds %d threds \n ",pi,run_time,j);
}
}	  
