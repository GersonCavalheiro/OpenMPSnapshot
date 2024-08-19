#include <stdio.h>
#include <stdlib.h>
#pragma omp target device(smp)
#pragma omp task out(a[n])
void generator(int *a, int n)
{
fprintf(stderr, "%s: a -> %p | a[%d] -> %p\n", __FUNCTION__, a, n, &a[n]);
a[n] = n;
}
#pragma omp target device(smp)
#pragma omp task in(*a)
void consumer(int *a, int n)
{
fprintf(stderr, "%s a[%d] -> %p\n", __FUNCTION__, n, &a[n]);
if (*a != n)
{
fprintf(stderr, "%d != %d\n", *a, n);
abort();
}
}
#define SIZE 10
int k[SIZE] = { 0 };
int main(int argc, char* argv[])
{
int i;
int *p;
for (i = 0; i < SIZE; i++)
{
generator(k, i); 
p = &k[i];
consumer(p, i);
}
#pragma omp taskwait
}
