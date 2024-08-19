#include <stdlib.h>
template <typename _T>
struct A
{
void f(_T t)
{
t = 1;
#pragma omp parallel
{
g();
t = 0;
}
if (t != 0)
abort();
}
void g() { }
};
int main(int argc, char *argv[])
{
A<int> a;
a.f(3);
A<float> b;
b.f(3.2f);
return 0;
}
