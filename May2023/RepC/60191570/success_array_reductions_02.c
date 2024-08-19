#include<assert.h>
void f(int n, int * v)
{
#pragma omp task reduction(+: [n]v)
{
int i;
for(i = 0; i < n; ++i)
v[i]++;
}
#pragma omp task reduction(+: [n]v)
{
int i;
for(i = 0; i < n; ++i)
v[i]++;
}
}
int main()
{
int i, v[10] = {0};
f(10, v);
#pragma omp taskwait
for(i = 0; i < 10; ++i)
assert(v[i] == 2);
return 0;
}
