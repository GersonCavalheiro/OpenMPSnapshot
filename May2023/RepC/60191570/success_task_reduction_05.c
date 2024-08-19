#include<assert.h>
#include<stdio.h>
#define N 5
int power(int base, int exp)
{
int result = 1;
for (int i = 0; i < exp; ++i)
{
result *= base;
}
return result;
}
int global_var;
int backtrack(int n)
{
if (n == 0)
{
return 1;
}
else
{
for (int i = 0; i < n; ++i)
{
#ifdef __NANOS6__
#pragma omp task weakreduction(+: global_var)  firstprivate(i)
#else
#pragma omp task reduction(+: global_var)  firstprivate(i)
#endif
{
int res = backtrack(i);
#ifdef __NANOS6__
#pragma omp task reduction(+: global_var)
#endif
global_var += res;
}
}
#pragma omp taskwait
return 0;
}
}
int main()
{
global_var = 0;
backtrack(N);
#pragma omp taskwait
int x = global_var;
assert(x == power(2,N-1));
}
