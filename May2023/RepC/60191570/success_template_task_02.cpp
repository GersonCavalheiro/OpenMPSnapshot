#include<assert.h>
#pragma omp task inout(*v)
template < typename t >
void producer(int* v);
template < typename S >
void producer(int* x)
{
*x = ((S) (*x));
}
template < typename P >
void consumer(int* v);
#pragma omp task in(*w)
template < typename Q >
void consumer(int* w);
template < typename R>
void consumer(int* x)
{
}
int main()
{
int b = -1;
producer<bool>(&b);
consumer<bool>(&b);
#pragma omp taskwait
assert(b == 1);
}
