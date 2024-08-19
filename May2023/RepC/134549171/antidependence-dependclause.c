#include <stdio.h>
#include <omp.h>
int main()
{
int x = 1;
#pragma omp parallel
#pragma omp single
{
#pragma omp task shared(x) depend(in: x)
printf("x = %d\n", x);
#pragma omp task shared(x) depend(out: x)
x = 2;
}
return 0;
}
