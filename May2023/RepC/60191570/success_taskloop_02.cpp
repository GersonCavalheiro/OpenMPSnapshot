#include<assert.h>
#define N 100
#define MAX_GRAINSIZE 7
int main(int argc, char* argv[])
{
for (int x = 1; x <= MAX_GRAINSIZE; ++x)
{
int a[N+1] = {0};
#pragma omp taskloop grainsize(x) shared(a)
for (int i = N; i >= 0; --i)
a[i]++;
int j;
#pragma omp taskloop grainsize(x) shared(a)
for (j = N; j >= 0; --j)
a[j]++;
for (int i = 0; i < N+1; ++i)
assert(a[i] == 2);
}
return 0;
}
