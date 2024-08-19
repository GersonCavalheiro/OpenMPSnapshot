#include <stdlib.h>
#pragma omp target device(smp) copy_deps
#pragma omp task inout([n]v)
void f(int n, int *v)
{
int i;
for (i = 0; i < n; i++)
{
v[i]++;
}
}
int main(int argc, char *argv[])
{
int i;
int w[10];
for (i = 0; i < 10; i++)
w[i] = i;
f(10, w);
#pragma omp taskwait
for (i = 0; i < 10; i++)
{
if (w[i] != (i+1))
abort();
}
}
