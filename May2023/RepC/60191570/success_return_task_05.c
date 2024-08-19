#include<assert.h>
#define N 10
#pragma omp task
int foo()
{
return 1;
}
int main()
{
int accum = 0;
for(int i = 0; i < N; ++i) accum += foo();
#pragma omp taskwait
assert(accum == N);
}
