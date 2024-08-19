#include"assert.h"
int n = 1;
#pragma omp task
void f(void)
{
n++;
}
int main() 
{
f();
#pragma omp taskwait
assert(n == 2);
}
