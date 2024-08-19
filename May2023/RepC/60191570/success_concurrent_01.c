#include<assert.h>
int main()
{
int i, res = 0;
int v[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
for (i = 0; i < 10; ++i)
{
#pragma omp task concurrent(res) shared(v) firstprivate(i)
{
#pragma omp atomic
res += v[i];
}
}
#pragma omp taskwait
assert(res == 55);
}
