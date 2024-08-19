#include <stdio.h>
#include <omp.h>
#include <time.h>
static long num_steps= 100000;
double step;
void main () {
int i;
double x, pi, sum = 0.0; 
clock_t begin_time = clock();
step = 1.0/(double) num_steps;
#pragma omp parallel 
{
double x;
#pragma omp for reduction(+:sum)
for (i=0;i< num_steps; i++){
x = (i+0.5)*step;
sum = sum + 4.0/(1.0+x*x);
}
}
pi = step * sum;
clock_t end_time = clock();
printf("PI: %f\nTIME: %f\n", pi, (double)(end_time-begin_time)/CLOCKS_PER_SEC);
}
