#include <omp.h>
#include <stdio.h>
int main()
{
int m = 500, n = 500, p = 500, q = 500;
int a[500][500], b[500][500];
for (int i = 0; i < 500; i++)
{
for (int j = 0; j < 500; j++)
{
a[i][j] = i;
b[i][j] = j;
}
}
int c[500][500];
int i, j, k;
double start_time = omp_get_wtime();
omp_set_num_threads(6);
#pragma omp parallel shared(a, b, c) private(i, j, k)
{
#pragma omp for schedule(static)
for (i = 0; i < m; i++)
{
for (j = 0; j < n; j++)
{
a[i][j] = 0;
for (k = 0; k < p; k++)
a[i][j] += b[i][k] * c[k][j];
}
}
}
printf("Total time = %lf seconds\n", omp_get_wtime() - start_time);
return 0;
}