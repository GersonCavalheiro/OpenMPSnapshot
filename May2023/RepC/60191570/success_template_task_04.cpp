#include <unistd.h>
#include <assert.h>
#pragma omp task inout(v[0;N][0;M])
template <typename T, int N, int M>
void producer(T (&v)[N][M])
{
sleep(1);
for (int i = 0; i < N; ++i)
for (int j = 0; j < M; ++j)
v[i][j] += j + i * N;
}
int main()
{
const int n = 10;
const int m = 20;
int v[n][m];
#pragma omp task out(v)
{
for (int i = 0; i < n; ++i)
for (int j = 0; j < m; ++j)
v[i][j] = 0;
}
#pragma omp task in(v)
{
for (int i = 0; i < n; ++i)
for (int j = 0; j < m; ++j)
assert(v[i][j] == 0);
}
producer(v);
#pragma omp task in(v)
{
for (int i = 0; i < n; ++i)
for (int j = 0; j < m; ++j)
assert(v[i][j] == j + i * n);
}
producer(v);
#pragma omp task in(v)
{
for (int i = 0; i < n; ++i)
for (int j = 0; j < m; ++j)
assert(v[i][j] == 2*(j + i * n));
}
#pragma omp taskwait
}
