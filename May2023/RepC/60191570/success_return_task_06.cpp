#include<assert.h>
#pragma omp task
int foo()
{
return 1;
}
int main()
{
const int n = foo();
#pragma omp taskwait
assert(n == 1);
}
