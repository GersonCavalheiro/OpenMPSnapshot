#include<assert.h>
struct A 
{
int _n; 
A(int p) : _n(p)  {}
#pragma omp task if (_n > 3) inout(*b)
void f(int a, int *b)
{
*b = *b + a;
}
};
void g()
{ 
A a(3);
int c = 3;
int d = 10;
a.f(c, &d);
assert(d == 13);
a.f(c, &d);
assert(d == 16);
#pragma omp taskwait
}
int main() { g(); }
