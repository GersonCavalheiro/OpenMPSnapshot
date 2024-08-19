#include<assert.h>
#define N 10
struct A
{
bool _sign;
A() : _sign(false) {}
A(bool b) : _sign(b) {}
void operator() (int *v, int n)
{
#pragma omp for
for (int i = 0; i < n; ++i)
{
v[i] = i;
}
}
A operator -()
{
A aux;
#pragma omp task out(aux) in(*this)
{
aux = *this;
aux._sign = !aux._sign;
}
#pragma omp taskwait
return aux;
}
bool operator ==( const A & a)
{
bool res;
#pragma omp task out(res) in(*this, a)
{
res = (this->_sign == a._sign);
}
#pragma omp taskwait
return res;
}
};
int main()
{
int v[N];
A A_true(true);
A_true(v, N);
for (int i = 0; i < N; ++i)
{
assert(v[i] == i);
}
A A_false = -A_true;
assert(A_true._sign == true);
assert(A_false._sign == false);
assert(!(A_true == A_false));
}
