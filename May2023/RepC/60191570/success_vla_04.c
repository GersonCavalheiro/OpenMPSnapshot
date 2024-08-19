#include<assert.h>
void foo(long N, int (*A)[N])
{
#pragma omp task
{
int (*myA1)[N];
myA1 = A;
myA1[N-1][N-1] = 2;
}
#pragma omp taskwait
}
int main()
{
long N = 2;
int v[N][N];
v[N-1][N-1] = -1;
foo(N, v);
assert(v[N-1][N-1] == 2);
}
