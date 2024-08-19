#include<assert.h>
#pragma omp task in(n) out(*out)
int foo(int n, int* out)
{
*out = n;
}
int main()
{
int x = -1;
if (1)  foo(1, &x);
#pragma omp taskwait
assert(x == 1);
}
