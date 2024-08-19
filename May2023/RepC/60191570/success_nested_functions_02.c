#if !defined(__ICC) || (__ICC >= 1300)
#include <assert.h>
void h(void)
{
int orig_x, x, z;
orig_x = x = 1;
z = 2;
#pragma omp task inout(*y, z)
void f(int *y)
{
(*y) += z;
}
f(&x);
#pragma omp taskwait
assert(x == (orig_x + z));
}
int main(int argc, char *argv[])
{
h();
return 0;
}
#else
int main(int argc, char *argv[])
{
return 0;
}
#endif
