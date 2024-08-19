#include <stdio.h>
#include <time.h>
#include <omp.h>
static long num_steps= 100000;         
double step;
#define NUM_THREADS 2
int main (void){
double  pi;
clock_t begin_time = clock();
step = 1.0/(double) num_steps;
double nthreads;
omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
{
int i, id, nthrds;    
double x, sum;
id = omp_get_thread_num();
nthrds= omp_get_num_threads();
if (id == 0)   
nthreads= nthrds;   
id = omp_get_thread_num();
nthrds= omp_get_num_threads();
for (i=id, sum=0.0; i< num_steps; i=i+nthreads) {
x = (i+0.5)*step;
sum += 4.0/(1.0+x*x);
}
#pragma omp critical 
pi += sum * step;
}
clock_t end_time = clock();
printf("PI: %f\nTIME: %f\n", pi, (double)(end_time-begin_time)/CLOCKS_PER_SEC);
return 0;
}
