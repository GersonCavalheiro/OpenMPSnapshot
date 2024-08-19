
#include <math.h>
#include <omp.h>
#include <stdio.h>

int main()
{
double sum = 0.0;
double x = 0.0, y = 0.0;
int num_steps = 50000;
double step_width = 1.0 / (double)num_steps;
int i = 0;

double exec_time = 0.0;
double start = omp_get_wtime();

#pragma omp parallel for reduction(+:sum) schedule(static, 10000)
for (i = 0; i < num_steps; i++) {
x = (i + 0.5) * step_width;
y = 4.0 / (1.0 + x * x);
sum += step_width * y;
}

double end = omp_get_wtime();
exec_time = end - start;

printf("\nThe computed pi number is: %f\n", sum);
printf("\nThe execution time is: %lf miliseconds.\n", exec_time * 1000);

return 0;
}
