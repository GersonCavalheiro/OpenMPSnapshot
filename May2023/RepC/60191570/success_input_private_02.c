#include<assert.h>
int main()
{
int x[2] = {-1, -1};
int *y = x;
#pragma omp task inprivate(x) shared(y)
{
assert(y != x);
x[0]++;
x[1]++;
}
#pragma omp taskwait
assert(x[0] == -1);
assert(x[1] == -1);
}
