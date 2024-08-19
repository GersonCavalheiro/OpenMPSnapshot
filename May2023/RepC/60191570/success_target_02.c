#include <stdlib.h>
#include <stdio.h>
#define M 20
const int m = M;
int b[M];
int main (int argc, char *argv[])
{
int *a = b;
{
int k;
for (k = 0; k < m; k++)
{
a[k] = -(k + 1);
}
}
#pragma omp task output([m] a)
{
int k;
for (k = 0; k < m; k++)
{
a[k] = k;
}
}
#pragma omp taskwait
int i;
for (i = 0; i < m; i++)
{
if (a[i] != i)
{
fprintf(stderr, "a[%d] == %d != %d\n", i, a[i], i);
abort();
}
}
return 0;
}
