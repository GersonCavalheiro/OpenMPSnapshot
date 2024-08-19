#include <stdlib.h>
#define N 500
unsigned int a[N][N];
void  __attribute__((noinline,noclone))
foo (unsigned int n)
{
int i, j;
unsigned int sum = 1;
#pragma acc kernels copyin (a[0:n]) copy (sum)
{
for (i = 0; i < n; ++i)
for (j = 0; j < n; ++j)
sum += a[i][j];
}
if (sum != 5001)
abort ();
}
