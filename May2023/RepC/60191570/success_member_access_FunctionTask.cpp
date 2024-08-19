#include<assert.h>
struct B
{
int n;
};
struct A
{
int n;
B* b;
#pragma omp task inout(n,b->n)
void f()
{
n++;
b->n++;
}
};
int main()
{
A a;
B b;
a.n = 1;
a.b = &b;
b.n = 2;
assert(a.n == 1 && a.b->n == 2);
a.f();
#pragma omp taskwait
assert(a.n == 2 && a.b->n == 3);
}
