#include <omp.h>
#include <stdio.h>

long num_steps = 1000000;
double t1; 

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

double ans = omp_get_wtime() - start;
printf("Pi = %lf\n", pi);

return ans;
}

float partial_sum_calculate(int i, double step)
{
double x, sum;
x = (i + 0.5) * step;
sum = 4.0 / (1.0 + x * x);

return sum;
}

double parallelExecTime(int numThreads)
{
float Sum;
double step = 1.0 / (double)num_steps;
int nthrds;
omp_set_num_threads(numThreads);

double start = omp_get_wtime();

#pragma omp parallel
{
float partial_sum;
int i, id;
id = omp_get_thread_num();
nthrds = omp_get_num_threads();
for (i = id; i < num_steps; i += nthrds)
{
partial_sum = partial_sum_calculate(i, step);


#pragma omp atomic
Sum = Sum + partial_sum;
}
}

double ans = omp_get_wtime() - start;

printf("Number of Threads: %d\tTime = %lf\tPi = %lf\n", nthrds, ans, Sum * step);

return ans;
}

int main()
{
t1 = serialExecTime();

printf("Time taken for Uni-Processor execution: %lf seconds\n", t1);
int i;
printf("Max number of threads: %d\n", omp_get_max_threads());
for (i = 2; i < omp_get_max_threads(); ++i)
parallelExecTime(i);

return 0;
}
