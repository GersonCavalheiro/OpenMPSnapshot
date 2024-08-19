#include <stdio.h>
#include <omp.h>
int main()
{
int arr[] = {4, 16, 1, 2, 13, 17, 20, 5, 2, 13, 8, 6, 11, 18, 13, 14, 12, 18, 19, 5};
int n = sizeof(arr) / sizeof(arr[0]);
int x;
printf("Unsorted order\n");
for (x = 0; x < n; x++)
{
printf("%d ", arr[x]);
}
double itime = omp_get_wtime();
int i, j, t, min_idx;
for (i = 0; i < n; i++)
{
min_idx = i;
#pragma omp parallel for schedule(dynamic) private(j)
for (j = i; j < n; j++)
{
if (arr[j] < arr[min_idx])
{
min_idx = j;
}
}
t = arr[i];
arr[i] = arr[min_idx];
arr[min_idx] = t;
}
printf("\n\nSorted Order\n");
for (i = 0; i < n; i++)
{
printf("%d ", arr[i]);
}
double ftime = omp_get_wtime();
double timegap = ftime - itime;
printf("\n\nTime taken is is %f", timegap);
return 0;
}
