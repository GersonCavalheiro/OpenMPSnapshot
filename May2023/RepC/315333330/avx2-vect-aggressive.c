#include "avx2-check.h"
#define N 64
float a[N];
int c[N];
__attribute__ ((noinline)) int
foo ()
{
int i, res = 0;
#pragma omp simd safelen(8)
for (i=0; i<N; i++)
{
float t = a[i];
if (t > 0.0f & t < 1.0e+2f)
if (c[i] != 0)
res += 1;
}
return res;
}
__attribute__ ((noinline)) float
hundred ()
{
return 100.0f;
}
static void
avx2_test (void)
{
int i, res;
for (i=0; i<N; i++)
{
c[i] = i % 4;
if (i < N / 2)
a[i] = (float) (i + 1);
else
a[i] = (float) i + hundred ();
}
if (foo () != 24)
abort ();
}
