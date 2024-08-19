#include<assert.h>
#define N 100
int main(int argc, char*argv[])
{
float v[N];
float res;
int i;
#pragma omp parallel for shared(v)
for (i = 0; i < N; ++i)
v[i] = i + 1;
res = -1;
#pragma omp parallel for reduction(max: res) shared(v)
for(i = 0; i < N; ++i)
res = res < v[i] ? v[i] : res;
assert(res == (float) N);
return 0;
}
