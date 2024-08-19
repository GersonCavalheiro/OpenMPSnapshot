#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define lld long long int
lld serial_fib(lld n)
{
if (n < 2)
return n;
else
return serial_fib(n - 1) + serial_fib(n - 2);
}
lld parallel_fib(lld n)
{
lld i, j;
if (n < 2)
return n;
else
{
#pragma omp task
i = parallel_fib(n - 1);
#pragma omp task
j = parallel_fib(n - 2);
return i + j;
}
}
lld parallel_fib_taskwait(lld n)
{
lld i, j;
if (n < 2)
return n;
#pragma omp task shared(i)
i = parallel_fib_taskwait(n - 1);
#pragma omp task shared(j)
j = parallel_fib_taskwait(n - 2);
#pragma omp taskwait
return i + j;
}
void check_correctness(lld result, lld n)
{
lld error = result - serial_fib(n);
if (error)
printf("\t==> ERROR OBTAINED\n");
else
printf("\tCorrectness checked\n");
}
int main()
{
lld n, result;
double start_time, time_taken;
printf("Enter n:  ");
scanf("%lld", &n);
if (n < 1)
{
printf("ERROR: n must be greater than 1\n");
exit(1);
}
printf("\nFibonacci in Serial:\n");
start_time = omp_get_wtime();
result = serial_fib(n);
time_taken = omp_get_wtime() - start_time;
printf("\tResult:\t%lld\n", result);
printf("\tTime taken:\t%lf s\n", time_taken);
check_correctness(result, n);
printf("\nFibonacci in Parallel:\n");
start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
result = parallel_fib(n);
}
time_taken = omp_get_wtime() - start_time;
printf("\tResult:\t%lld\n", result);
printf("\tTime taken:\t%lf s\n", time_taken);
check_correctness(result, n);
printf("\nFibonacci in Parallel with tast-wait:\n");
start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
result = parallel_fib_taskwait(n);
}
time_taken = omp_get_wtime() - start_time;
printf("\tResult:\t%lld\n", result);
printf("\tTime taken:\t%lf s\n", time_taken);
check_correctness(result, n);
return 0;
}
