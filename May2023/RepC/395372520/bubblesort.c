#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
void swap(int *xp, int *yp)
{
int temp = *xp;
*xp = *yp;
*yp = temp;
}
void bubbleSort(int arr[], int n)
{
int j = 0;
int i = 0;
int first;
#pragma omp parallel for default(none),shared(arr,i,n,first)
for (i = 0; i < n-1; i++) {  
first = i % 2; 
#pragma omp parallel for default(none),shared(arr,first,n)
for (j = first; j < n-1; j++) {
if (arr[j] > arr[j+1]) {
swap(&arr[j], &arr[j+1]);
}
}
}
}
bool isArraySorted(int arr[], int size)
{
bool result = true;
for (int i=0; i < size - 1; i++)
if(arr[i] > arr[i+1]) 
result = false;
return result;
}
int main()
{
int i,n = 10000;
int *arr = (int*) malloc(n*sizeof(int));
#pragma omp parallel for
for(i=0; i < n; i++)
arr[i] = rand()%n;
bubbleSort(arr, n);
printf("Is array sorted?: \n");
fputs(isArraySorted(arr, n) ? "true\n" : "false\n", stdout);
return 0;
}