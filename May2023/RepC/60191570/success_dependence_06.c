#include<unistd.h>
#include<assert.h>
void f(int (*v)[10])
{
#pragma omp task inout(v[1][0:9])
{
sleep(1);
int k = 0;
for (k = 0; k < 10; ++k)
{
v[1][k] = 2;
}
}
int *v2 = &((*(v+1))[0]);
#pragma omp task in(([10]v2)[0:9])
{
int k = 0;
for (k = 0; k < 10; ++k)
{
assert(v2[k]==2);
}
}
#pragma omp taskwait
}
int main()
{
int x[10][10];
int i, j;
for (i  = 0; i < 10; ++i)
for (j = 0; j < 10; ++j)
x[i][j] = -1;
f(x);
return 0;
}
