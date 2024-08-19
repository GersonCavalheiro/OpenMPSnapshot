#include<assert.h>
int** global;
#pragma omp task
void f(int n, int v[n][n])
{
assert(global == (int**)v);
if (n > 0)
{
f(n-1, v);
#pragma omp taskwait
}
}
int main()
{
int n = 10;
int v[n][n];
global = (int**) v;
f(n, v);
#pragma omp taskwait
return 0;
}
