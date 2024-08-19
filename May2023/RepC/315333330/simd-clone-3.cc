#include "simd-clone-2.h"
#pragma omp declare simd notinbranch
int
S::f0 (int x)
{
return x + s;
}
#pragma omp declare simd notinbranch uniform(this)
int
S::f1 (int x)
{
return x + s;
}
#pragma omp declare simd notinbranch linear(this:sizeof(this)/sizeof(this))
int
S::f2 (int x)
{
return x + this->S::s;
}
#pragma omp declare simd uniform(this) aligned(this:32) linear(x)
int
T::f3 (int x)
{
return t[x];
}
