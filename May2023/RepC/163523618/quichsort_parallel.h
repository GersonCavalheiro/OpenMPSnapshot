#pragma once
#include <omp.h>
#include <stdlib.h>
inline void swap_(int *x, int *y)
{
int temp = *x;
*x = *y;
*y = temp;
}
inline int choose_pivot_(int i, int j)
{
return((i + j) / 2);
}
inline int partition(int* arr, int left, int right,int pivot)
{
swap_(&arr[left], &arr[pivot]);
int key = arr[left];
int left_cursor = left + 1;
int right_cursor = right;
while (left_cursor <= right_cursor)
{
while ((left_cursor <= right) && (arr[left_cursor] <= key))
{
left_cursor++;
}
while ((right_cursor >= left) && (arr[(right_cursor)] > key))
{
(right_cursor)--;
}
if (left_cursor < right_cursor)
swap_(&arr[left_cursor], &arr[(right_cursor)]);
}
swap_(&arr[left], &arr[(right_cursor)]);
return right_cursor;
}
int* add_scan_serial(int* arr, int size)
{
int* scanned = (int*)malloc(size * sizeof(int));
scanned[0] = arr[0];
for (int i = 1; i < size; ++i)
{
int x = scanned[i - 1] + arr[i];
scanned[i] = x;
}
return scanned;
}
int reduction(int* arr, int size)
{
if (size == 0)
{
return 0;
}
int reduced_sum = arr[0];
for (int i = 1; i < size; ++i)
{
reduced_sum += arr[i - 1] + arr[i];		
}
return reduced_sum;
}
int*  reduced_para()
{
int A[] = { 84, 30, 95, 94, 36, 73, 52, 23, 2, 13 };
static int S[10] = { 0 };
#pragma omp parallel
{
int n;
#pragma omp for reduction(+:S[:10])
for ( n = 0; n < 10; ++n) {
for (int m = 0; m <= n; ++m) {
S[n] += A[m];
}
}
}
return S;
}
int* add_scan(int* arr, int size)
{
if (size == 1)
{
return arr;
}
int* arr_to_scan = (int*)malloc(size * sizeof(int));
int* arr_even_idx;
if (size%2 != 0)
{
arr_even_idx = (int*)malloc((size / 2 + 1) * sizeof(int));
arr_even_idx[size / 2] = size - 1;
}
else { arr_even_idx = (int*)malloc(size / 2 * sizeof(int)); }
arr_to_scan[0] = arr[0];
for (int i = 1; i < size; ++i)
{
arr_to_scan[i] = arr[i] + arr_to_scan[i - 1];
}
arr_to_scan = add_scan(arr_to_scan, size);
return arr_to_scan;
}
int* paarallel_partition(int* arr, int size, int pivot)
{
int* arr_flags = (int*)malloc(size * sizeof(int));
for (int i = 0; i < size; ++i)
{
arr_flags[i] = arr[i] <= arr[pivot] ? 1 : 0;
}
return arr_flags;
}
inline void quicksort_parallel(int arr[], int left, int right)
{
if (left < right)
{
int pivot = choose_pivot_(left, right); 
int partition_pt =  partition(arr, left, right, pivot);
#pragma omp parallel 
quicksort_parallel(arr, left, partition_pt - 1);
#pragma omp parallel 
quicksort_parallel(arr, partition_pt + 1, right);
}
}
