
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
static long num_steps = 100000000;
#define PAD 8
#define NUM_THREADS 61
double step;
int main (int argc, char* argv[])
{
if(argc < 2){
printf("Usage: ./a.out [number of threads]\n");
exit(0);
}

unsigned int specified_threads = atoi(argv[1]);
omp_set_num_threads(specified_threads);
double sum[NUM_THREADS][PAD];
int nthreads;
double start_time, run_time;

step = 1.0/(double) num_steps;

start_time = omp_get_wtime();

#pragma omp parallel
{
int i;
double x;
int id = omp_get_thread_num();
int num_threads =  omp_get_num_threads();   	 
if(id == 0)
nthreads = num_threads;
for (i=id, sum[id][0] = 0.0; i< num_steps; i+=num_threads){
x = (i+0.5)*step;
sum[id][0] = sum[id][0] + 4.0/(1.0+x*x);
}
}

int i;
double pi =0.0;
for(i = 0; i < nthreads;i++)
pi += sum[i][0] * step;

run_time = omp_get_wtime() - start_time;
printf("NUMBER OF THREADS: %d\n", nthreads);
printf("\n pi with %ld steps is %lf in %lf seconds\n ",num_steps,pi,run_time);
return 0;
}	  

