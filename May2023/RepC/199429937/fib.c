#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define lld long long int
lld fib_serial(lld n)
{
lld i, j;
if (n < 2)
return n;
else
{
i = fib_serial(n - 1);
j = fib_serial(n - 2);
return i + j;
}
}
lld fib_parallel(lld n)
{
lld i, j;
if (n < 2)
return n;
else
{
i = fib_parallel(n - 1);
j = fib_parallel(n - 2);
return i + j;
}
}
lld fib_parallel_taskwait(lld n)
{
lld i, j;
if (n < 20)
return fib_serial(n);
else
{
#pragma omp task
i = fib_parallel_taskwait(n - 1);
#pragma omp task
j = fib_parallel_taskwait(n - 2);
#pragma omp taskwait
return i + j;
}
}
lld check_correctness(lld num, lld n)
{
if (n < 2)
return num - n;
lld i, x1, x2, x3;
x1 = 0;
x2 = 1;
for (i = 2; i <= n; ++i)
{
x3 = x1 + x2;
x1 = x2;
x2 = x3;
}
return num - x3;
}
int main()
{
lld n, answer, error;
double start_time, time_taken;
printf("Enter the nth fibonacci number to print\n");
scanf("%lld", &n);
start_time = omp_get_wtime();
answer = fib_serial(n);
time_taken = omp_get_wtime() - start_time;
error = check_correctness(answer, n);
printf("nth fibonacci number is %lld\n", answer);
printf("Time taken for serial approach is %lf s\n", time_taken);
if (error)
printf("Error obtained: %lld\n", error);
else
printf("The output obtained is correct and has no errors.\n");
printf("\n");
start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
answer = fib_parallel(n);
}
time_taken = omp_get_wtime() - start_time;
error = check_correctness(answer, n);
printf("The nth fibonacci number is %lld\n", answer);
printf("Time taken for parallel approach is %lf s\n", time_taken);
if (error)
printf("Error obtained: %lld\n", error);
else
printf("The output obtained is correct and has no errors.\n");
printf("\n");
start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp single
answer = fib_parallel_taskwait(n);
}
time_taken = omp_get_wtime() - start_time;
error = check_correctness(answer, n);
printf("The nth fibonacci number is %lld\n", answer);
printf("Time taken for parallel approach with taskwait is %lf s\n", time_taken);
if (error)
printf("Error obtained: %lld\n", error);
else
printf("The output obtained is correct and has no errors.\n");
printf("\n");
return 0;
}