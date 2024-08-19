#include <stdlib.h>
#include <string.h>
#define N 1024
int a[N];
int main(int argc, char *argv[])
{
int i;
int BS = 64;
memset(a, 0, sizeof(a));
#pragma omp task shared(a)
{
#pragma omp for
for (i = 0; i < N; i += BS)
{
int j;
for (j = i; j < i+ BS; j++)
{
a[j] = i;
}
}
}
#pragma omp taskwait
for (i = 0; i < N; i += BS)
{
int j;
for (j = i; j < i+ BS; j++)
{
if (a[j] != i) abort();
}
}
return 0;
}
