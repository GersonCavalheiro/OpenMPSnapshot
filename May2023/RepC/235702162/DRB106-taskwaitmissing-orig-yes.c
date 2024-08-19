#include <stdio.h>
unsigned int input = 10;
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
int res= i+j; 
#pragma omp taskwait
return res;
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
printf ("Fib(%d)=%d (correct answer should be 55)\n", input, result);
return 0;
}
