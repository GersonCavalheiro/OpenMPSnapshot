#include<stdio.h>
#include<assert.h>
#define N 10
int main()
{
int res = 0;
int v[N][N];
for (int i = 0; i < N; ++i)
for (int j= 0; j < N; ++j)
v[i][j] = i*N + j+1;
for (int i = 0; i < N; ++i)
{
#ifdef __NANOS6__
#pragma oss task weakreduction(+: res) in(v) firstprivate(i)
#else
#pragma oss task reduction(+: res) in(v) firstprivate(i)
#endif
{
#ifdef __NANOS6__
#pragma oss task reduction(+: res) in(v) firstprivate(i)
#endif
res  += v[i][0];
for (int j = 0+1; j < N; ++j)
{
#pragma omp task reduction(+: res) in(v) firstprivate(i)
{
res += v[i][j];
}
}
#ifndef __NANOS6__
#pragma omp taskwait
#endif
}
}
#pragma omp task in(res)
{
assert(res == (( (N*N) * ((N*N)+1)) / 2));
}
#pragma omp taskwait
}
