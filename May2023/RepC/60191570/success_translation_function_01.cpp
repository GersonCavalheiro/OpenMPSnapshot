#include <assert.h>
template <typename T>
void f()
{
int a[10];
for (int i = 0; i < 10; i++)
a[i] = i;
#pragma omp target device(smp) copy_deps
#pragma omp task inout([10]a)
{
for (int i = 0; i < 10; i++)
a[i]++;
}
#pragma omp taskwait
for (int i = 0; i < 10; i++)
assert(a[i] == (i + 1));
}
int main(int argc, char *argv[])
{
f<int>();
return 0;
}
