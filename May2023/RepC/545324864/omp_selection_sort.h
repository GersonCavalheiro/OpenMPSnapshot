#ifndef OMP_SELECTION_SORT
#define OMP_SELECTION_SORT
#include <omp.h>
#include "../utils/utils_sort.h"
#ifndef OMP_THREADS
#define OMP_THREADS 2
#endif
void omp_selection_sort(int arr[], int n)
{
int i, j, minimum;
for (i = 0; i < n - 1; i++)
{
minimum = i;
#pragma omp parallel for schedule(static) 
for (j = i + 1; j < n; j++)
{
if (arr[minimum] > arr[j]){
minimum = j;
}
}
if (minimum != i){
swap(&arr[minimum], &arr[i]);
}
}
}
#endif