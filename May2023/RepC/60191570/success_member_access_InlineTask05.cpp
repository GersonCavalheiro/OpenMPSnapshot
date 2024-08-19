#include<assert.h>
struct A
{
int m,n;
A(int _n, int _m) : n(_n), m(_m) { }
void foo(int &a, int &b)
{
int d = 0;
#pragma omp task inout(a,b) 
{
d = 1;
a+=d;
b+=d;
n+=d;
m+=d;
}
#pragma omp taskwait
assert(d == 0);
}
};
void foo()
{
A a(1, 2);
int n1 = 3, m1 = 4;
a.foo(n1, m1);
assert(n1 == 4);
assert(m1 == 5);
assert(a.n == 2);
assert(a.m == 3);
}
int main() 
{
foo();    
}
