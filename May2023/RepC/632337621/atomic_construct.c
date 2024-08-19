#include <stdio.h>
#include <omp.h>
int main()
{
int x[5] = {1, 0, 0, 0, 0};
int n = 5;
printf("\nInitial Declaration:\n");
for (int i = 0; i < n; i++)
{
printf("%d ", x[i]);
}
printf("\n");
#pragma omp parallel for shared(x)
for (int i = 0; i < n; i++)
{
printf("Thread %d - i = %d\n", omp_get_thread_num(), i);
#pragma omp atomic write
x[i] = x[0] * 2 * i;
}
printf("\nArray after construct: \n");
for (int i = 0; i < n; i++)
{
printf("%d ", x[i]);
}
return 0;
}
