#include<assert.h>
#define N 10
void init(int (*v)[N])
{
for (int i = 0; i < N; ++i)
for (int j = 0; j < N; ++j)
v[i][j] = 0;
}
void check(int (*v)[N])
{
for (int i = 0; i < N; ++i)
for (int j = 0; j < N; ++j)
assert(v[i][j] == 1);
}
int main(int argc, char* argv[])
{
int v[N][N];
int i, j;
{
init(v);
#pragma omp parallel for collapse(2)
for(i = 0; i < N; ++i)
for(j = 0; j < N; ++j)
v[i][j] += 1;
check(v);
}
{
init(v);
#pragma omp parallel
{
#pragma omp for collapse(2)
for(i = 0; i < N; ++i)
for(j = 0; j < N; ++j)
v[i][j] += 1;
}
check(v);
}
{
init(v);
#pragma omp parallel
{
#pragma omp single
{
int SIZE = N;
#pragma omp taskloop grainsize(SIZE) collapse(2)
for(i = 0; i < N; ++i)
for(j = 0; j < N; ++j)
v[i][j] += 1;
}
}
check(v);
}
return 0;
}
