#include<assert.h>
struct A
{
int v;
A()
{
#pragma omp task
{
v = 2;
}
}
};
int main()
{
A a;
#pragma omp taskwait
assert(a.v == 2);
}
