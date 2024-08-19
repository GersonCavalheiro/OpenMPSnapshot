#include "pr68762.h"
#pragma omp declare simd
double
baz (double x)
{
return x;
}
double
fn (double x)
{
return foo (x);
}
