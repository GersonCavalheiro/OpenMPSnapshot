#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include <omp.h>
#define MAX 1000
void merge(int arr[], int l, int m, int r)
{
int i, j, k;
int n1 = m - l + 1;
int n2 =  r - m;
int L[n1], R[n2];
for (i = 0; i < n1; i++)
L[i] = arr[l + i];
for (j = 0; j < n2; j++)
R[j] = arr[m + 1+ j];
i = 0; 
j = 0; 
k = l; 
while (i < n1 && j < n2)
{
if (L[i] <= R[j])
{
arr[k] = L[i];
i++;
}
else
{
arr[k] = R[j];
j++;
}
k++;
}
while (i < n1)
{
arr[k] = L[i];
i++;
k++;
}
while (j < n2)
{
arr[k] = R[j];
j++;
k++;
}
}
void mergeSort(int arr[], int l, int r)
{
if (l < r)
{
int m = l+(r-l)/2;
#pragma omp parallel sections num_threads(2) 
{
#pragma omp section 
mergeSort(arr, l, m);
#pragma omp section
mergeSort(arr, m+1, r);
}
merge(arr, l, m, r);
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
int arr[MAX],i;
srand(time(NULL));
for(i=0;i<MAX;i++){
arr[i] = rand() % MAX;
}
clock_t t1,t2;
t1 = clock();
mergeSort(arr, 0, MAX - 1);
t2 = clock();
double t_time = (double)(t2-t1)/CLOCKS_PER_SEC;
printf("Sorted array: \n");
printArray(arr, MAX);
printf("\t\tTime Elapsed: %.5f\n",t_time);
return 0;
}
