#include<assert.h>
#pragma omp task
int fact(int n)
{
if ( n == 0 || n == 1) return 1;
return fact(n-1) * n; 
}
int main()
{
int x = fact(6);
#pragma omp taskwait
assert(x == 720);
}
