#ifndef OMP_QUICK_SORT
#define OMP_QUICK_SORT
#include <omp.h>
#include "../utils/utils_sort.h"
#include "../sorting/quick_sort.h"
#include "../sorting/bubble_sort.h"
#ifndef OMP_THREADS
#define OMP_THREADS 2
#endif
int chunksize;
int omp_assemble_partition(int arr[], int low, int high)
{
int pivot = arr[high];
int i = low - 1;
for (int j = low; j < high; j++)
{
if (arr[j] <= pivot)
{
i++;
swap(&arr[i], &arr[j]);
}
}
i++;
swap(&arr[i], &arr[high]);
return i;
}
void omp_quick_sort_task(int arr[], int p1, int p2)
{
if (p1 >= p2 || p1 < 0)
{
return;
}
int diff = p2 - p1;
if (diff < 8)
{
bubble_sort(arr, diff);
}
int pt = omp_assemble_partition(arr, p1, p2);
#pragma omp task shared(arr) if (p2 - p1 > chunksize)
omp_quick_sort_task(arr, p1, pt - 1);
#pragma omp task shared(arr) if (p2 - p1 > chunksize)
omp_quick_sort_task(arr, pt + 1, p2);
}
void omp_quick_sort(int arr[], int p1, int p2, int size)
{
chunksize = size / OMP_THREADS;
#pragma omp parallel shared(arr)
{
#pragma omp single nowait
{
omp_quick_sort_task(arr, p1, p2);
}
}
}
#endif