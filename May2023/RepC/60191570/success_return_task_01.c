#include<assert.h>
#pragma omp task
int foo()
{
return 1;
}
int main()
{
int x = -1;
x = foo();
#pragma omp taskwait
assert(x == 1);
}
