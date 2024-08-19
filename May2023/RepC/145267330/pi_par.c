

#include <stdio.h>
#include <omp.h>
static long num_steps = 100000000;
double step;
int main ()
{
int i, nprocs;
double x, pi, sum = 0.0;
double start_time, run_time;

step = 1.0/(double) num_steps;

nprocs=100;

for (i=1;i<=nprocs;i++){
sum = 0.0;
omp_set_num_threads(i);
start_time = omp_get_wtime();
#pragma omp parallel  
{
#pragma omp for reduction(+:sum) private(x)
for (i=1;i<= num_steps; i++){
x = (i-0.5)*step;
sum = sum + 4.0/(1.0+x*x);
}
}

pi = step * sum;
run_time = omp_get_wtime() - start_time;
printf("%f\t%d\n", run_time, i);
}
}	  


