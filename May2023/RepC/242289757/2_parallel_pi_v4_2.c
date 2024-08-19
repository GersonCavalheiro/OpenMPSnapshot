

#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 12
static long long num_steps = 1000000000;
double step;
int main()
{
int i;
double x;
double pi, sum = 0.0;
double start_time, run_time;
step = 1.0 / (double)num_steps;
omp_set_num_threads(NUM_THREADS);
start_time = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum) private(x, i)
{
for (i = 1; i <= num_steps; i++)
{
x = (i - 0.5) * step;		
sum += 4.0 / (1.0 + x * x); 
}
}

pi = step * sum;
run_time = omp_get_wtime() - start_time;
printf("\n pi with %lld steps is %lf in %lf seconds\n ", num_steps, pi, run_time);
}
