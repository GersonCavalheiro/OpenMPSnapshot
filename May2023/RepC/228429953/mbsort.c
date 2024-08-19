#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
void print(int a[], int n)
{
int i;
for(i = 0; i < n; ++i)
printf("%d ", a[i]);
printf("\n");
}
void swap(int *a, int *b)
{
int temp = *a;
*a = *b;
*b = temp;
}
void sbsort(int a[], int n)
{
int i, j;
double t1 = omp_get_wtime();
for(i = 0; i < n - 1; ++i)
for(j = 0; j < n - i -1; ++j)
if(a[j] > a[j + 1])
swap(&a[j], &a[j + 1]);
double t2 = omp_get_wtime();
print(a, n);
printf("\nTime taken: %f seconds.\n", t2 - t1);
}
void pbsort(int a[], int n)
{
int i, j, key;
double t1 = omp_get_wtime();
for(i = 0; i < n - 1; ++i)
{
key = i % 2;
#pragma omp parallel for
for(j = key; j < n - 1; ++j)
if(a[j] > a[j + 1])
swap(&a[j], &a[j + 1]);
}
double t2 = omp_get_wtime();
print(a, n);
printf("\nTime taken: %f seconds.\n\n", t2 - t1);
}
int main(void)
{
int n, i;
printf("Enter the size of the array: ");
scanf("%d", &n);
int a[n], b[n];
srand(time(0));
for(i = 0; i < n; ++i)
{
a[i] = rand();		
b[i] = a[i];	
}
printf("\nLet the array's elements be:\n\n");
print(a, n);
printf("\n\nElements sorted by the bubble sort algorithm:\n\n");
sbsort(a, n);
printf("\n\nElements sorted by the odd-even sort algorithm:\n\n");
pbsort(b, n);	
return 0;	
}
