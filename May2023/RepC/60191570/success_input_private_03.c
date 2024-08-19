#include<assert.h>
int main()
{
int x[2][2] = {{-1, -1}, {-1, -1}};
int (*y)[2] = x;
#pragma omp task inprivate(x)
{
assert(y != x);
x[0][0]++;
x[0][1]++;
x[1][0]++;
x[1][1]++;
}
#pragma omp taskwait
assert(x[0][0] == -1);
assert(x[0][1] == -1);
assert(x[1][0] == -1);
assert(x[1][1] == -1);
}
