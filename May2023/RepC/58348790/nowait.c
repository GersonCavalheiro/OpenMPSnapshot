#include <stdio.h>
#define SIZE 5
void test(int *a, int *b, int *c, int size) 
{
int i;
#pragma omp parallel
{
#pragma omp for nowait
for (i = 0; i < size; i++)
b[i] = a[i] * a[i];
#pragma omp for nowait
for (i = 0; i < size; i++)
c[i] = a[i]/2;
}
}
int main( ) 
{
int a[SIZE], b[SIZE], c[SIZE];
int i;
for (i=0; i<SIZE; i++)
a[i] = i;
test(a,b,c, SIZE);
for (i=0; i<SIZE; i++)
printf("%d, %d, %d\n", a[i], b[i], c[i]);
}
