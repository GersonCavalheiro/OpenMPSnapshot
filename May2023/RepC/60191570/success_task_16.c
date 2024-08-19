#include<assert.h>
#include<omp.h>
#define N 1000
__thread int res = 0;
int main()
{
int result = 0;
int i;
int v[N];
for (i = 0; i < N; ++i) v[i] = i+1;
for (i = 0; i < N; ++i)
{
#pragma omp task shared(v) firstprivate(i)
{
res += v[i];
}
}
#pragma omp taskwait
#pragma omp for reduction(+:result)
for (i = 0; i < omp_get_max_threads(); ++i)
{
result += res;
}
assert(result == ( ( N * (N+1) ) /2) );
}
