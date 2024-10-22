#if !defined(__hppa__) || !defined(__hpux__)
#include <complex.h>
#endif
#define FRAC (1.0 / (1LL << 48))
typedef double _Complex Type;
int close_enough (Type a, Type b)
{
Type diff = a - b;
double mag2_a = __real__(a) * __real__ (a) + __imag__ (a) * __imag__ (a);
double mag2_diff = (__real__(diff) * __real__ (diff)
+ __imag__ (diff) * __imag__ (diff));
return mag2_diff / mag2_a < (FRAC * FRAC);
}
#define N 100
static int __attribute__ ((noinline))
vector (Type ary[N], Type sum, Type prod)
{
Type tsum = 0, tprod = 1;
#pragma acc parallel vector_length(32) copyin(ary[0:N])
{
#pragma acc loop vector reduction(+:tsum) reduction (*:tprod)
for (int ix = 0; ix < N; ix++)
{
tsum += ary[ix];
tprod *= ary[ix];
}
}
if (!close_enough (sum, tsum))
return 1;
if (!close_enough (prod, tprod))
return 1;
return 0;
}
static int __attribute__ ((noinline))
worker (Type ary[N], Type sum, Type prod)
{
Type tsum = 0, tprod = 1;
#pragma acc parallel num_workers(32) copyin(ary[0:N])
{
#pragma acc loop worker reduction(+:tsum) reduction (*:tprod)
for (int ix = 0; ix < N; ix++)
{
tsum += ary[ix];
tprod *= ary[ix];
}
}
if (!close_enough (sum, tsum))
return 1;
if (!close_enough (prod, tprod))
return 1;
return 0;
}
static int __attribute__ ((noinline))
gang (Type ary[N], Type sum, Type prod)
{
Type tsum = 0, tprod = 1;
#pragma acc parallel num_gangs (32) copyin(ary[0:N])
{
#pragma acc loop gang reduction(+:tsum) reduction (*:tprod)
for (int ix = 0; ix < N; ix++)
{
tsum += ary[ix];
tprod *= ary[ix];
}
}
if (!close_enough (sum, tsum))
return 1;
if (!close_enough (prod, tprod))
return 1;
return 0;
}
int main (void)
{
Type ary[N], sum = 0, prod = 1;
for (int ix = 0; ix < N;  ix++)
{
double frac = ix * (1.0 / 1024) + 1.0;
ary[ix] = frac + frac * 2.0i - 1.0i;
sum += ary[ix];
prod *= ary[ix];
}
if (vector (ary, sum, prod))
return 1;
if (worker (ary, sum, prod))
return 1;
if (gang (ary, sum, prod))
return 1;
return 0;
}
