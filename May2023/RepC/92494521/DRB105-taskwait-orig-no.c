#include <stdio.h>
#include <assert.h>
unsigned int input = 30;
int fib(unsigned int n)
{
if (n<2)
return n;
else
{
int i, j;
#pragma omp task shared(i)
i=fib(n-1);
#pragma omp task shared(j)
j=fib(n-2);
#pragma omp taskwait
return i+j;
}
}
int main()
{
int result = 0;
#pragma omp parallel
{
#pragma omp single
{
result = fib(input);
}
}
printf ("Fib(%d)=%d\n", input, result);
assert (result==832040);
return 0;
}
