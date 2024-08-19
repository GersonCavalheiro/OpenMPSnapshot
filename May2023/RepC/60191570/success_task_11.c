#include<assert.h>
void foo(int*copy)
{
#pragma omp task inout([1]copy)
{
assert((*copy) == 0);
*copy = 1;
}
#pragma omp task inout(*copy)
{
assert((*copy) == 1);
*copy = 2;
}
}
int main()
{
int res = 0;
foo(&res);
#pragma omp taskwait
assert(res == 2);
return 0;
}
