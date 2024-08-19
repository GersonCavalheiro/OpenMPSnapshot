#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define lld long long int
lld check_correctness(lld arr[], lld n);
void swap(lld *a, lld *b);
void printArray(lld arr[], lld size);
void quickSort(lld arr[], lld low, lld high);
void quickSort_parallel(lld arr[], lld low, lld high);
int main()
{
double start_time, time_taken;
lld i, n, input, error;
printf("Enter array size: ");
scanf("%lld", &n);
lld arr[n], arr2[n];
for (i = 0; i < n; ++i)
{
arr[i] = n - i;
arr2[i] = n - i;
}
start_time = omp_get_wtime();
quickSort(arr, 0, n - 1);
time_taken = omp_get_wtime() - start_time;
error = check_correctness(arr, n);
printf("Time taken for serial approach is %lf s\n", time_taken);
if (error)
printf("Error obtained\n");
else
printf("Checked for correctness\n");
printf("\n");
start_time = omp_get_wtime();
quickSort_parallel(arr2, 0, n - 1);
time_taken = omp_get_wtime() - start_time;
error = check_correctness(arr2, n);
printf("Time taken for parallel approach is %lf s\n", time_taken);
if (error)
printf("Error obtained\n");
else
printf("Checked for correctness\n");
printf("\n");
return 0;
}
void swap(lld *a, lld *b)
{
lld t = *a;
*a = *b;
*b = t;
}
void printArray(lld arr[], lld size)
{
lld i;
for (i = 0; i < size; i++)
printf("%lld ", arr[i]);
printf("\n");
}
lld partition(lld arr[], lld low, lld high)
{
lld pivot = arr[high]; 
lld i = (low - 1);	 
for (lld j = low; j <= high - 1; j++)
{
if (arr[j] < pivot)
{
i++; 
swap(&arr[i], &arr[j]);
}
}
swap(&arr[i + 1], &arr[high]);
return (i + 1);
}
void quickSort(lld arr[], lld low, lld high)
{
if (low < high)
{
lld pi = partition(arr, low, high);
quickSort(arr, low, pi - 1);
quickSort(arr, pi + 1, high);
}
}
void quickSort_parallel(lld arr[], lld low, lld high)
{
if (low < high)
{
lld pi;
#pragma omp task shared(pi)
pi = partition(arr, low, high);
#pragma omp taskwait
quickSort_parallel(arr, low, pi - 1);
printf("%d\n", omp_get_thread_num());
#pragma omp taskwait
quickSort_parallel(arr, pi + 1, high);
printf("%d\n", omp_get_thread_num());
}
}
lld check_correctness(lld arr[], lld n)
{
lld i, error = 0;
for (i = 0; i < n - 1; ++i)
{
if (arr[i + 1] < arr[i])
return 1;
}
return 0;
}
