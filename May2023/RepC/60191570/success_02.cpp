#include<assert.h>
void foo(int* n)
{
*n = 2;
}
#pragma omp task inout(*n) final(1)
void bar(int* n)
{
*n = 4;
}
int main()
{
int x = -1;
#pragma omp task inout(x) final(1)
{
foo(&x);
}
#pragma omp taskwait on(x)
assert(x == 2);
bar(&x);
#pragma omp taskwait on(x)
assert(x == 4);
}
