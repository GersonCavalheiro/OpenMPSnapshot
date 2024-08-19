#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
void swap(int* a, int* b)
{
int t = *a;
*a = *b;
*b = t;
}
int partition (int arr[], int low, int high)
{
int pivot = arr[high];    
int i = (low - 1);  
for (int j = low; j <= high- 1; j++)
{
if (arr[j] <= pivot)
{
i++;    
swap(&arr[i], &arr[j]);
}
}
swap(&arr[i + 1], &arr[high]);
return (i + 1);
}
void quickSort(int arr[], int low, int high, int aux)
{
if (low < high)
{
int pi = partition(arr, low, high);
if(aux > 20)
{
quickSort(arr, low, pi - 1, aux);
quickSort(arr, pi + 1, high, aux);
return;
}
#pragma omp parallel sections
{
#pragma omp section
{
quickSort(arr, low, pi - 1, aux+1);
}
#pragma omp section
{
quickSort(arr, pi + 1, high, aux+1);
}
}
}
}
void printArray(int arr[], int size)
{
int i;
for (i=0; i < size; i++)
printf("%d ", arr[i]);
printf("\n");
}
int main()
{
int i,n = 10000000;
int *arr = (int*) malloc(n*sizeof(int));
for(i=0; i < n; i++)
arr[i] = rand()%n;
quickSort(arr, 0, n-1, 0);
return 0;
}
