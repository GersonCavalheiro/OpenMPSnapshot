#include <assert.h>
#pragma omp task  inout([1] var)
void f(int * var, int cnst)
{
int i = 0;
double x = 1;
for (i = 0; i < 10000; ++i)
{
x = x * 2.0;
}
assert(*var == cnst);
(*var) = (*var) + 1;
}
#pragma omp task  concurrent([1] var)
void g(int * var)
{
(*var) = 0;
}
int main()
{
int i;
int result = 0;
int *ptrResult = &result;
for (i = 0; i < 10; i++)
f(ptrResult, i);
g(ptrResult);
for (i = 0; i < 10; i++)
f(ptrResult, i);
#pragma omp taskwait
}
