#include<assert.h>
struct C
{
int n;
};
struct B
{
C * c;
int n;
};
struct A
{
int n;
B* b;
#pragma omp task inout(n,b->n, b->c->n)
void f()
{
n++;
b->n++;
b->c->n++;
}
};
int main()
{
A a;
B b;
C c;
c.n = 1;
b.n = 2;
b.c = &c;
a.n = 3;
a.b = &b;
a.f();
#pragma omp taskwait
assert(a.n == 4);
assert(a.b->n == 3);
assert(a.b->c->n == 2);
}
