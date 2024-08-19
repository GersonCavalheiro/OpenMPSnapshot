#include<assert.h>
struct A
{
#pragma omp task
int foo() { return 1; }
};
struct B
{
#pragma omp task
int foo() { return 1; }
};
int main()
{
A a;
B b;
int y = a.foo() + b.foo();
#pragma omp taskwait
}
