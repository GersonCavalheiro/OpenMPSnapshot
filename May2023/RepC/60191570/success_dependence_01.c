#include<unistd.h>
#include<assert.h>
#define N 10
#define M 100
struct A
{
int x[M];
};
int main()
{
struct A a[N];
int i, j;
for (i = 0; i < N; ++i)
{
for (j = 0; j < M; ++j)
{
a[i].x[j] = -1;
}
}
int z = 3, ub = 10, lb = 0;
int *p = &(a[z].x[lb]);
#pragma omp task out(a[z].x[lb:ub])
{
sleep(1);
int k;
for (k = lb; k <= ub; k++)
a[z].x[k] = 2;
}
#pragma omp task in(*p)
{
int i;
for (i = 0; i <= (ub - lb); i++)
assert(p[i] == 2);
}
#pragma omp taskwait
return 0;
}
