#include<stdio.h>
#include<assert.h>
#define N 1000
int main()
{
int res = 0;
int v[N];
for (int i = 0; i < N; ++i) v[i] = i+1;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(+:res) in(v) firstprivate(i)
{
int tmp = res;
tmp += v[i];
res = tmp;
}
}
#pragma omp task in(res)
{
assert(res == ( ( N * (N+1) ) /2) );
}
#pragma omp taskwait
}
