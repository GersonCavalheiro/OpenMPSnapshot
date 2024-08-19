#include <stdlib.h>
#define N 500
unsigned int a[N][N];
void  __attribute__((noinline,noclone))
foo (void)
{
int i, j;
unsigned int sum = 1;
#pragma acc kernels copyin (a[0:N]) copy (sum)
{
for (i = 0; i < N; ++i)
for (j = 0; j < N; ++j)
sum += a[i][j];
}
if (sum != 5001)
abort ();
}
