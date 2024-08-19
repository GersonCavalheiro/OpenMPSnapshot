#include<assert.h>
#define N 5
#pragma omp task
int f(int i)
{
return i + 1;
}
int main()
{
int i, x = 0;
for (i = 0; i < N; ++i)
{
x += f(i) + i;
}
#pragma omp taskwait
assert(x == 25);
}
