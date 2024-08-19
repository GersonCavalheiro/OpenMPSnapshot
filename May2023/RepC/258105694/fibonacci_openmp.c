#include <stdio.h>
#include <math.h>
#include <omp.h>
unsigned long long fib(int n)
{
unsigned long long i, j;
if (n<2)
return n;
else
{
#pragma omp task shared(i) firstprivate(n)
i=fib(n-1);
#pragma omp task shared(j) firstprivate(n)
j=fib(n-2);
#pragma omp taskwait
return i+j;
}
}
int
main()
{
int n = 40;
omp_set_dynamic(0);
omp_set_num_threads(4);
int i;
#pragma omp parallel shared(n)
{
#pragma omp for ordered schedule(static,5)
for(i=0;i<n;i++)
#pragma omp ordered
printf ("fib(%d) = %llu\n",i, fib(i));
}
return 0;
}
