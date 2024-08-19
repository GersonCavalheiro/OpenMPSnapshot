#include <stdio.h>
#include <omp.h>
int main()
{
int x[2][5] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}};
int count = 0, num = 0;
printf("\nInitial Declaration:\n");
for (int i = 0; i < 2; i++)
{
for (int j = 0; j < 5; j++)
{
printf("%d ", x[i][j]);
}
printf("\n");
}
printf("\n");
#pragma omp parallel for collapse(2)
#pragma omp reduction(+:count)
for (int i = 0; i < 2; i++)
{
for (int j = 0; j < 5; j++)
{
printf("\ni = %d, j = %d from thread-%d", i, i, omp_get_thread_num());
if (x[i][j] % 2 == 0)
{
count++;
}
}
}
#pragma omp master
{
printf("\nMaster Thread:\nDone till %dth Array , Even nums count = %d\n", num, count);
}
printf("\nEven nums count = %d\n", count);
return 0;
}
