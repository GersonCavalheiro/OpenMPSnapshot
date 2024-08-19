#include <stdio.h>
#include <string.h>
#include <assert.h>
void foo(int n)
{
int v[n][n];
memset(v, 0, sizeof(v));
#pragma omp for firstprivate(v)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
{
assert(v[i][j] == 0);
v[i][j]++;
}
}
#pragma omp for shared(v)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
{
assert(v[i][j] == 0);
v[i][j]++;
}
}
#pragma omp for firstprivate(v)
for (int i = 0; i < n; ++i)
{
for (int j = 0; j < n; ++j)
{
assert(v[i][j] == 1);
v[i][j]++;
}
}
}
int main()
{
foo(10);
return 0;
}
