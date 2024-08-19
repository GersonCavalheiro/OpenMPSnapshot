
#include "stdafx.h"
#include "StopWatch.h"
#include <omp.h>

int Fibonacci(int n)
{
int x, y;
if (n < 2)
{
return n;
}
else
{
x = Fibonacci(n - 1);
y = Fibonacci(n - 2);
return (x + y);
}
}
int FibonacciTask(int n)
{
int x, y;
if (n < 2) return n;
else
{
#pragma omp task shared(x)
x = Fibonacci(n - 1);
#pragma omp task shared(y)
y = Fibonacci(n - 2);
#pragma omp taskwait
return (x + y);
}
}
int _tmain(int argc, _TCHAR* argv[])
{

const int MAX = 40;
int FibNumber[MAX] = { 0 };
int i = 0;
omp_set_num_threads(5);

CStopWatch watch;

watch.Start();
#pragma omp parallel
{
#pragma omp for
for (i = 1; i < MAX; i++)
{
FibNumber[i] = Fibonacci(i);
}
}
watch.End();
printf("Parallel for ҿð : %f\n", watch.GetDurationMilliSecond());
watch.Start();
#pragma omp parallel
{
#pragma omp single private(i)
for (i = 1; i < MAX; i++)
{
#pragma omp task
FibNumber[i] = Fibonacci(i);
}
}
watch.End();
printf("task 1 ҿð : %f\n", watch.GetDurationMilliSecond());
watch.Start();
#pragma omp parallel
{
#pragma omp single private(i)
for (i = 1; i < MAX; i++)
{
FibNumber[i] = FibonacciTask(i);
}
}
watch.End();

printf("task 2 ҿð : %f\n", watch.GetDurationMilliSecond());
printf("Ǻġ  : ");
for (i = 1; i < MAX; i++)
printf("%d ", FibNumber[i]);
printf("\n");
return 0;
}
