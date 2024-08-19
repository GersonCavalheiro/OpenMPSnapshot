#include<assert.h>
#pragma omp task
int foo()
{
return 2;
}
#pragma omp task
int bar()
{
int x = (1) ? foo() : 1;
#pragma omp taskwait on(x)
return x;
}
int main()
{
int x = (1) ? bar() : 0;
#pragma omp taskwait
assert(x == 2);
}
