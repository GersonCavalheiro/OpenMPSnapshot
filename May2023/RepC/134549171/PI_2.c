#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
static long num_steps = 100000000;
double step;
int main (int argc , char ** argv)
{
unsigned int num_threads = atoi(argv[1]);
int nthreads;
omp_set_num_threads(num_threads);
int i,j;
double pi, full_sum = 0.0;
double start_time, run_time;
double * sum = (double *)malloc((num_threads + 1) * sizeof(double));
step = 1.0/(double) num_steps;
full_sum=0.0;
start_time = omp_get_wtime();
#pragma omp parallel 
{
int i;
int id = omp_get_thread_num();
int numthreads = omp_get_num_threads();
double x;
sum[id] = 0.0;
if (id == 0) {
nthreads = numthreads;
}
for (i=id;i< num_steps; i+=numthreads){
x = (i+0.5)*step;
sum[id] = sum[id] + 4.0/(1.0+x*x);
}
}
for(full_sum = 0.0, i=0;i<nthreads;i++)
full_sum += sum[i];
pi = step * full_sum;
run_time = omp_get_wtime() - start_time;
printf("%lf\n",run_time);
}	  
