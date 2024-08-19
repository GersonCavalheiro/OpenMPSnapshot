#include"assert.h"
struct A 
{
int n;
A(int _n) : n(_n) {}
#pragma omp task
void f(int m)
{
n++;
}
};
int main() 
{
A a(1);
a.f(1);
#pragma omp taskwait
assert(a.n == 2);
}
