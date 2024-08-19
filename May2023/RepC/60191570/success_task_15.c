#include<assert.h>
const int n = 1000;
int main()
{
int x1 = -1;
int x2 = -1;
#pragma omp task shared(x1, x2)
{
#pragma omp for lastprivate(x1)
for (x1 = 0; x1 < n; ++x1)
{
}
#pragma omp for
for (x2 = 0; x2 < n; ++x2)
{
}
}
#pragma omp taskwait
assert(x1 == n);
assert(x2 == -1);
return 0;
}
