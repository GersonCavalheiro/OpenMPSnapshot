#include <stdlib.h>
void f(int *x, int n)
{
int i;
for (i = 0; i < n; i++)
{
#pragma omp task depend(out : x[i]) firstprivate(i)
{
x[i] = i;
}
#pragma omp task depend(inout : x[i]) firstprivate(i)
{
if (x[i] != i)
{
abort();
}
x[i]++;
}
#pragma omp task depend(in : x[i]) firstprivate(i)
{
if (x[i] != (i+1))
{
abort();
}
}
}
#pragma omp taskwait
}
int main(int argc, char *argv[])
{
#pragma omp parallel
{
#pragma omp single
{
int c[100];
f(c, 100);
}
}
return 0;
}
