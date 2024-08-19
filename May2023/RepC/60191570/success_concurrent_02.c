#include<assert.h>
#define N 10
int main()
{
int i;
int v[N], res = 0;
for ( i = 0; i < N; ++i) v[i] = i;
for (int  i = 0; i < N; ++i)
{
#pragma omp task shared(v) concurrent(res) no_copy_deps
#pragma omp atomic
res += v[i];
}
#pragma omp taskwait
assert(res == (((N - 1) * (N + 1 - 1)) / 2));
}
