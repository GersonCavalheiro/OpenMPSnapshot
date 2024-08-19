#include <assert.h>
template <typename T>
void f(T* t, T lower, T length, T val)
{
T i;
#pragma omp parallel for
for (i = lower; i < (lower + length); i++)
{
t[i] = val + i;
}
}
int main(int argc, char *argv[])
{
int a[10];
f(a, 0, 10, 42);
int i;
for (i = 0; i < 10; i++)
{
assert(a[i] == 42 + i);
}
return 0;
}
