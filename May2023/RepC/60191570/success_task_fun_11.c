#include <stdlib.h>
#pragma omp target device(smp) copy_deps
#pragma omp task inout(a[0;10])
void f(int *a)
{
int i;
for (i = 0; i < 10; i++)
a[i]++;
}
int main(int argc, char *argv[])
{
int v[10];
int i;
for (i = 0; i < 10; i++)
v[i] = i;
f(v);
#pragma omp taskwait
for (i = 0; i < 10; i++)
{
if (v[i] != (i+1))
abort();
}
return 0;
}
