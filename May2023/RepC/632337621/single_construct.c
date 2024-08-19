#include <stdio.h>
#include <omp.h>
int main()
{
int x[5] = {1, 2, 3, 4, 5};
int target = 4;
int ind = -1;
printf("\nInitial Declaration:\n");
for (int i = 0; i < 5; i++)
{
printf("%d ", x[i]);
}
#pragma omp parallel shared(x)
{
#pragma omp single
printf("\nStarting search with thread-%d...\n", omp_get_thread_num());
#pragma omp for
for (int i = 0; i < 5; i++)
{
if (x[i] == target)
{
ind = i;
}
}
#pragma omp single
printf("\nTarget at Index %d with thread-%d", ind, omp_get_thread_num());
}
return 0;
}
