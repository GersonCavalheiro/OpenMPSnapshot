#include<assert.h>
#pragma omp task inout((*a)[1:4])
void foo(int (*a)[5])
{
int i;
for (i = 1; i < 5; ++i)
{
(*a)[i] = i;
}
}
int main()
{
int x[5] = {-1, -1, -1, -1, -1};
foo(&x);
#pragma omp taskwait
assert(x[0] == -1);
int i;
for (i = 1; i < 5; ++i)
{
assert(x[i] == i);
}
}
