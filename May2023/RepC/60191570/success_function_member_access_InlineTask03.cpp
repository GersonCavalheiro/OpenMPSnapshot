#include<assert.h>
struct A
{
int m,n;
A(int _n, int _m) : n(_n), m(_m) { }
void foo(int &a, int &b)
{
#pragma omp task
{
a++;
b++;
n++;
m++;
}
}
void bar(int &n1, int &m1)
{
foo(n1, m1);
}
};
void foo()
{
A a(1, 2);
int n1 = 3, m1 = 4;
a.bar(n1, m1);
#pragma omp taskwait
assert(n1 == 3);
assert(m1 == 4);
assert(a.n == 2);
assert(a.m == 3);
}
int main() 
{
foo();    
}
