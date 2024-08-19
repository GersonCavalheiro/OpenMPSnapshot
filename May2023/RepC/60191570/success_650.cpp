#include <cstdio>
void foo(int N)
{
#pragma omp parallel for
for (int i = 0; i < N; i++)
{
printf("%d\n", i);
}
}
void foo2(int N)
{
#pragma omp parallel for
for (int i(0); i < N; i++)
{
printf("%d\n", i);
}
}
