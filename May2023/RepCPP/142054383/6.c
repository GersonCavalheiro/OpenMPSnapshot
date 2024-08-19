#include <omp.h>
#include <stdio.h>

long num_steps = 100000;

double serialExecTime()
{
double step;

int i;
double x, pi, sum = 0.0;
step = 1.0 / (double)num_steps;

double start = omp_get_wtime();

for (i = 0; i < num_steps; i++)
{
x = (i + 0.5) * step;
sum = sum + 4.0 / (1.0 + x * x);
}
pi = step * sum;


printf("Serial Execution Results: Pi = %lf\n", pi);

double ans = omp_get_wtime() - start;

return ans;
}

float partial_sum_calculate(int i, double step)
{
double x, sum;
x = (i + 0.5) * step;
sum = 4.0 / (1.0 + x * x);

return sum;
}

double parallelExecTime()
{
float Sum;
double step = 1.0 / (double)num_steps;

double start = omp_get_wtime();

int nthrds, i;
#pragma omp parallel for reduction(+ : Sum)

for (i = 0; i < num_steps; i++)
{
float partial_sum = partial_sum_calculate(i, step);
Sum = Sum + partial_sum;
}

nthrds = omp_get_num_threads();


double ans = omp_get_wtime() - start;
printf("Multi-threaded Result: Pi = %lf\n", Sum * step);
return ans;
}

int main()
{
printf("Time taken for Uni-Processor execution: %lf seconds\n", serialExecTime());

printf("Time taken for Multi-Processor execution: %lf seconds\n", parallelExecTime());

return 0;
}
