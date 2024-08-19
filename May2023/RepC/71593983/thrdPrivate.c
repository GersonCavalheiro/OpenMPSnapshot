#include<stdio.h>
#include<omp.h>
int main()
{
static double x = 5.0;
#pragma omp threadprivate(x)
#pragma omp parallel copyin(A) 
{
#pragma omp single
A = 5;
#pragma omp single
printf("A = %i \n", A);
}
#pragma omp parallel copyin(A)
{
#pragma omp single
printf("A = %i \n", A);
}
return 0;
}
