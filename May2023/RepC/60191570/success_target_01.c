#include <stdio.h>
#include <stdlib.h>
const int N = 10;
void hi (int M, int *a)
{
int k;
for (k = 0; k < M; k++)
{
a[k] = k;
}
}
int main (int argc, char *argv[])
{
int i;
int a[N];
for (i = 0; i < N; i++) 
{
a[i] = -(i + 1);
}
for (i = 0; i < N; i++) 
{
#pragma omp target device(smp)
#pragma omp task firstprivate(N) inout(a)
{
hi(N, a);
}
}
#pragma omp taskwait
for (i = 0; i < N; i++)
{
if (a[i] != i)
{
fprintf(stderr, "a[%d] == %d != %d\n", i, a[i], i);
abort();
}
}
return 0;
}
