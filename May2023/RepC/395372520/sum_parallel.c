#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main()
{
double sum   =  0;
int width    =  40000000;
#pragma omp parallel for simd reduction(+:sum)
for(int i = 0; i < width; i++)
{
sum += i;
}
printf("\nSum = %lf\n",sum);
}