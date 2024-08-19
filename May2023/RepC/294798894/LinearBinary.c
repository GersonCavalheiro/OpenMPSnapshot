#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
clock_t t, t2;
double cpu_time_used, cpu_time_used2;
int linearSearch(int* A, int n, int tos);
int main(){
int number, iter =0, find;
int* Arr;
int s[number];
printf("Done by Maitreyee\n\n");
Arr = (int *)malloc( number * sizeof(int));
scanf("%d", &number);
for(; iter<number; iter++){
scanf("%d", &Arr[iter]);
}
scanf("%d", &find);
printf("\nTo find: %d\n", find);
t = clock();
int indexx = linearSearch(Arr, number, find);
t = clock()-t;
printf("\nLinear Search :\n");
if(indexx == -1){
printf("Not found");
}
else
printf("Found at %d\n", indexx);
cpu_time_used = ((double)t)/CLOCKS_PER_SEC;
printf("\nTime taken for linear search: %f", cpu_time_used);
t2 = clock();
int index2 = binarysearch(Arr, number, find);
t2 = clock()-t;
printf("\nBinary Search: \n");
if(index2 == -1){
printf("Not found");
}
else
printf("Found at %d\n", index2);
cpu_time_used2 = ((double)t2)/CLOCKS_PER_SEC;
printf("\nTime taken for binary search: %f", cpu_time_used2);
return 0;
}
int linearSearch(int* A, int n, int tos){
int mid = n/2;
int foundat = -1;
int one_quart = mid/2;
int three_quart = 3*one_quart;
double start_time;
start_time = omp_get_wtime();
int loc[4];
double end_time;
#pragma omp parallel sections
{
#pragma omp section
{
start_time = omp_get_wtime();
loc[0] = linear(A, t, 0, one_quart);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",1, end_time);
}
#pragma omp section
{
start_time = omp_get_wtime();
loc[1] = linear(A, t, one_quart+1, mid);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",1, end_time);
}
#pragma omp section
{
start_time = omp_get_wtime();
loc[2] = linear(A, t, mid+1, three_quart);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",3, end_time);
}
#pragma omp section
{
start_time = omp_get_wtime();
loc[3] = linear(A, t, three_quart+1, n-1);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",4, end_time);
}
}
int i;
for(int i = 0; i<4; i++)
{
if(loc[i]!=-1)
{
return loc[i];
}
}
}
int linear(int A[], int x, int low, int high)
{
int i, foundat;
for(int i =low ; i< high; i++){
if(A[i] == x)
foundat = i+1;
return foundat;
}
return -1;
}
int binarysearch(int* A, int n, int k)
{
int mid = n/2;
double start_time;
int low = 0;
int high = n -1;
int one_quart = mid/2;
int three_quart = 3*one_quart;
start_time = omp_get_wtime();
double end_time;
int loc[4];
#pragma omp parallel sections
{
#pragma omp section
{
start_time = omp_get_wtime();
loc[0] = binary(A, t, 0, one_quart);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",1, end_time);
}
#pragma omp section
{
start_time = omp_get_wtime();
loc[1] = binary(A, t, one_quart+1, mid);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",1, end_time);
}
#pragma omp section
{
start_time = omp_get_wtime();
loc[2] = binary(A, t, mid+1, three_quart);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",3, end_time);
}
#pragma omp section
{
start_time = omp_get_wtime();
loc[3] = binary(A, t, three_quart+1, n-1);
end_time = omp_get_wtime()-start_time;
printf("Time for section %d: \t %f \n",4, end_time);
}
}
int i;
for(int i = 0; i<4; i++)
{
if(loc[i]!=-1)
{
return loc[i]+1;
}
}
}
int binary(int arr[], int x, int low, int high)
{
while(low <= high)
{
int mid = low + (high-low)/2;
if(arr[mid] == x)
return mid;
if(arr[mid] < x)
low = mid + 1;
else
high = mid - 1;
}
return -1;
}
