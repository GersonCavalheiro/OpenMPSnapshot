#include<stdio.h>
#include<omp.h>
static long num_steps=100000;
double step;
#define NUM_THREADS 50
int main(){
int i; double x, pi, sum = 0.0;
step = 1.0/(double) num_steps;
omp_set_num_threads(NUM_THREADS);
#pragma omp parallel 
{
#pragma omp for reduction(+:sum)        
for (i=0; i<num_steps; i++){
x = (i+0.5)*step;
sum +=  4.0 / (1.0 + x*x);
}   
}
pi += sum * step;
printf("%lf",pi);
}