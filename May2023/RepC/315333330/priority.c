#include <omp.h>
#include <stdlib.h>
#define N 64
int main()
{
int tsknum=0, prio[N];
int max_priority = omp_get_max_task_priority ();
int saved_tsknum = -1;
int i;
#pragma omp parallel num_threads(1)
#pragma omp single private (i)
{
for (i = 0; i < N; i++)
#pragma omp task priority(i ^ 1)
{
int t;
#pragma omp atomic capture seq_cst
t = tsknum++;
prio[t] = i ^ 1;
}
#pragma omp atomic read seq_cst
saved_tsknum = tsknum;
}
if (saved_tsknum == 0)
{
for (i = 0; i < N; i++)
if (i < N - max_priority)
{
if (prio[i] < max_priority)
abort ();
}
else if (i != N - prio[i] - 1)
abort ();
}
return 0;
}
