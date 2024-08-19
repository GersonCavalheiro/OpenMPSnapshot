#pragma once
void swap(int *x, int *y)
{
int temp = *x;
*x = *y;
*y = temp;
}
inline int choose_pivot(int i, int j)
{
return((i + j) / 2);
}
inline void quicksort_serial(int arr[], int left, int right)
{
int key, i, j, pivot;
if (left < right)
{
pivot = choose_pivot(left, right); 
swap(&arr[left], &arr[pivot]);
key = arr[left];
i = left + 1;
j = right;
while (i <= j)
{
while ((i <= right) && (arr[i] <= key))
i++;
while ((j >= left) && (arr[j] > key))
j--;
if (i < j)
swap(&arr[i], &arr[j]);
}
swap(&arr[left], &arr[j]);
quicksort_serial(arr, left, j - 1);
quicksort_serial(arr, j + 1, right);
}
}