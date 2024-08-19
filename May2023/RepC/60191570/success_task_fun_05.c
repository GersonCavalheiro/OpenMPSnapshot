#include <assert.h>
#pragma omp task  concurrent([1] var)
void f(int * var, int cnst)
{
assert(*var == cnst);
}
#pragma omp task  inout([1] var)
void g(int * var)
{
(*var) = 2;
}
int main()
{
int i;
int result = 1;
int *ptrResult = &result;
for (i = 0; i < 10; i++)
f(ptrResult, 1);
g(ptrResult);
for (i = 0; i < 10; i++)
f(ptrResult, 2);
#pragma omp taskwait
}
