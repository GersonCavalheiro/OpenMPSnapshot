#include <stdlib.h>
template <typename _T>
struct A
{
void f(_T t);
void g() { }
};
template <typename _Q>
void A<_Q>::f(_Q q)
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
A<int> a;
a.f(3);
A<float> b;
b.f(3.2f);
return 0;
}
