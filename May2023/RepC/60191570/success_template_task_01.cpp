#include<assert.h>
#pragma omp task out(*x)
template < typename T >
void producer(T* x)
{
*x = (T) 2;
}
#pragma omp task in(*x)
template < typename T >
void consumer(T* x)
{
}
int main()
{
int x = -1;
producer(&x);
consumer(&x);
#pragma omp taskwait
assert(x == 2);
}
