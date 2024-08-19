#include<assert.h>
int main()
{
int res = 0;
int v[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
#pragma omp task shared(v, res)
{
int i;
#pragma omp for
for (i = 0; i < 10; ++i)
{
#pragma omp atomic
res += v[i];
}
}
#pragma omp taskwait
assert(res == 45);
}
