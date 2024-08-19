#include <stdlib.h>
template <typename _S>
struct A
{
template <typename _T>
void f(_T t);
void g() { }
};
template <typename _F>
template <typename _Q>
void A<_F>::f(_Q q)
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
a.f(3.2f);
A<float> b;
b.f(3);
b.f(3.2f);
return 0;
}
