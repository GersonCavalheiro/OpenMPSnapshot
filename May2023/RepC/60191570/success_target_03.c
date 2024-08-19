#include <stdlib.h>
#include <stdio.h>
#define n 20
const int M = n;
int b[n];
int main (int argc, char *argv[])
{
int *a = b;
{
int k;
for (k = 0; k < M; k++)
{
a[k] = k + 1;
}
}
#pragma omp target device(smp) copy_in(a[0:M-1])
#pragma omp task firstprivate(stderr) firstprivate(M)
{
int k;
for (k = 0; k < M; k++)
{
if (a[k] != (k + 1))
{
fprintf(stderr, "a[%d] == %d != %d\n", k, a[k], k+1);
abort();
}
}
}
#pragma omp taskwait
return 0;
}
