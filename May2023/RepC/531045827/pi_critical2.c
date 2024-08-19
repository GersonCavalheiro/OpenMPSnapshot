#include<stdio.h>
#include<omp.h>
static long num_steps=100000;
double step;
#define NUM_THREADS 50
int main(){
double pi;
step = 1.0/(double) num_steps;
omp_set_num_threads(NUM_THREADS);
#pragma omp parallel 
{
int i, id, nthrds;
double x, sum;
id = omp_get_thread_num();
nthrds = omp_get_num_threads();
for (i=id, sum=0.0; i<num_steps; i=i+nthrds){
x = (i+0.5)*step;
#pragma omp critical 
pi +=  4.0 / (1.0 + x*x);
}
}
pi = pi * step;
printf("%lf",pi);
}
