#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define SIZE 10
int main() 
{
int i;
int max;
int a[SIZE] = {1,2,3,4,5,6,7,8,9,10};
max = a[0];
#pragma omp parallel 
{
#pragma omp for  
for (i = 1; i < SIZE; i++) {
if (a[i] > max) {
#pragma omp critical
{
if (a[i] > max)
max = a[i];
}
}
}
}
printf("max = %d\n", max);
}
