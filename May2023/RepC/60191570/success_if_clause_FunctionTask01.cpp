#include<assert.h>
#pragma omp task if (a > 3) inout(*b)
void f(int a, int *b)
{
*b = *b + a;
}
void g()
{ 
int c = 3;
int d = 10;
f(c, &d);
assert(d == 13);
f(c, &d); 
assert(d == 16);
#pragma omp taskwait
}
int main() { g(); }
