#include<assert.h>
#pragma omp task in(x)
int foo(int x)
{
return x + 1;
}
int main()
{
int valor = 2;
int x = 0;
x = foo(valor) + 1 + foo(valor);
#pragma omp taskwait
assert(x == 7);
}
