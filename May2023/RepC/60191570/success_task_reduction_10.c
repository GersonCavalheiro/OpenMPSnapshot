#include <assert.h>
int main()
{
int x = 0;
int xA[10] = {};
#pragma omp task reduction(+: x) reduction(+: xA)
{
x++;
for (int i = 0; i < 10; ++i)
{
xA[i]++;
}
}
#pragma omp taskwait
assert(xA[1] == 1);
}
