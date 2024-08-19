#include <stdlib.h>
struct A
{
template <typename _T>
void f(_T t);
void g() { }
};
template <typename _Q>
void A::f(_Q q)
{
q = 1;
#pragma omp parallel
{
g();
q = 0;
}
if (q != 0)
abort();
}
int main(int argc, char *argv[])
{
A a;
a.f(3);
A b;
b.f(3.2f);
return 0;
}
