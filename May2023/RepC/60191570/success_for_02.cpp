#include <assert.h>
template < typename T>
void foo(T& x, int n)
{
int i;
#pragma omp for schedule(static) shared(x)
for (i = 0; i <= n; ++i)
{
#pragma omp atomic
x += i;
}
}
int main()
{
int x = 0;
int n = 1000;
foo(x, n);
assert(x == (((n + 1)* n) / 2));
}
