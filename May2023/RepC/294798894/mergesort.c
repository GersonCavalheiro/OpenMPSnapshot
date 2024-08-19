#include<stdio.h>
#include "omp.h"
void merge(int a[], int l, int m, int r)
{
int temp[m-l+1], temp2[r-m];
for(int i=0; i<(m-l+1); i++)
temp[i]=a[l+i];
for(int i=0; i<(r-m); i++)
temp2[i]=a[m+1+i];
int i=0, j=0, k=l;
while(i<(m-l+1) && j<(r-m))
{
if(temp[i]<temp2[j])
a[k++]=temp[i++];
else
a[k++]=temp2[j++];
}
while(i<(m-l+1))
a[k++]=temp[i++];
while(j<(r-m))
a[k++]=temp2[j++];
}
void mergeSort(int a[], int l, int r)
{
if(l<r)
{
int m=(l+r)/2;
#pragma omp parallel sections num_threads(2)
{
#pragma omp section
{
mergeSort(a,l,m);
}
#pragma omp section
{
mergeSort(a,m+1,r);
}
}
merge(a,l,m,r);
}
}
void print(int a[], int n)
{
for(int i=0; i<n; i++)
printf("%d , ", a[i]);
printf("\n");
}
int main()
{
int n;
printf("Done by Maitreyee\n\n");
double t1,t2;
t1 = omp_get_wtime ( );
printf("Enter the number of random numbers required: ");
scanf("%d",&n);
int a[n];
for(int i=0; i<n; i++)
a[i] = rand();
printf("Unsorted array; ");
print(a,n);
mergeSort(a,0,n-1);
printf("\nSorted array: ");
print(a,n);
t1 = omp_get_wtime ( )-t1;
printf("Time for execution: %8f ", t1);
return 0;
}
