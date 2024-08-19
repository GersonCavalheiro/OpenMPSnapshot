#include<assert.h>
int main()
{
int x = 1;
int *y = &x;
#pragma omp task inprivate(x) shared(y)
{
assert(&x != y);
x++;
}
#pragma omp taskwait
assert(x == 1);
}
