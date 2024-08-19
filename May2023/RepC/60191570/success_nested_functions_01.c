#if !defined(__ICC) || (__ICC >= 1300)
#include <stdlib.h>
void f1(void)
{
void g(int *x)
{
(*x)++;
}
int y;
y = 1;
#pragma omp task inout(y)
{
g(&y);
}
#pragma omp taskwait
if (y != 2) abort();
}
void f2(void)
{
#pragma omp task inout(*x)
void g(int *x)
{
(*x)++;
}
int y;
y = 1;
g(&y);
#pragma omp taskwait
if (y != 2) abort();
}
int main(int argc, char *argv[])
{
f1();
f2();
return 0;
}
#else
int main(int argc, char *argv[])
{
return 0;
}
#endif
