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
int backtrack(int n)
{
if (n == 0)
{
return 1;
}
else
{
int local_counter = 0;
for (int i = 0; i < n; ++i)
{
#pragma omp task reduction(+: local_counter)  firstprivate(i)
{
int res = backtrack(i);
local_counter += res;
}
}
#pragma omp taskwait
return local_counter;
}
}
int main()
{
int x = backtrack(N);
assert(x == power(2,N-1));
}
