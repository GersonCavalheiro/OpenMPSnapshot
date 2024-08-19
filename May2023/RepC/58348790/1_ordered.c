#include <stdio.h>
#include <omp.h>
static float a[1000], b[1000], c[1000];
int main( ) 
{
int i;
#pragma omp parallel
{
#pragma omp for ordered
for (i = 0 ; i < 5 ; i++)
#pragma omp ordered
printf("test2() iteration %d\n", i);
}
}
