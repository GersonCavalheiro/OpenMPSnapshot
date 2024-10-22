#include <stdio.h>
#include <omp.h>
int main()
{
int mat1[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int mat2[3][3] = {{4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
int sum[3][3];
double itime = omp_get_wtime();
printf("\nMatrix Declarations:\nMatrix 1\n");
for (int i = 0; i < 3; i++)
{
for (int j = 0; j < 3; j++)
{
printf("%d ", mat1[i][j]);
}
printf("\n");
}
printf("Matrix 2\n");
for (int i = 0; i < 3; i++)
{
for (int j = 0; j < 3; j++)
{
printf("%d ", mat2[i][j]);
}
printf("\n");
}
omp_set_num_threads(4);
#pragma omp parallel for collapse(2)
for (int i = 0; i < 3; i++)
{
for (int j = 0; j < 3; j++)
{
sum[i][j] = mat1[i][j] + mat2[i][j];
printf("Thread %d - sum[%d][%d]=%d\n",
omp_get_thread_num(), i, j, sum[i][j]);
}
}
printf("\nSum of Matrices\n");
for (int i = 0; i < 3; i++)
{
for (int j = 0; j < 3; j++)
{
printf("%d ", sum[i][j]);
}
printf("\n");
}
double ftime = omp_get_wtime();
double timetaken = ftime - itime;
printf("\nTime Taken = %f", timetaken);
return 0;
}
