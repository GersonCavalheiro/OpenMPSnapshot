#include<assert.h>
#define N 100
#define MAX_GRAINSIZE 7
int main(int argc, char* argv[])
{
for (int x = 1; x <= MAX_GRAINSIZE; ++x)
{
int a[N];
for (int i = 0; i < N; ++i) a[i] = -1;
#pragma omp taskloop grainsize(x) out(a[i]) nogroup
for (int i = 0; i < N; ++i)
a[i] = 0;
#pragma omp taskloop grainsize(x) inout(a[i])
for (int i = 0; i < N; ++i)
{
assert(a[i] == 0);
a[i]++;
}
for (int i = 0; i < N; ++i)
assert(a[i] == 1);
}
return 0;
}
