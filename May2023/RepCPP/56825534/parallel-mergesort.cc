

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "sort.hh"

#define THRESHOLD 1250000

void mergeSort(keytype *A, int size, keytype *temp);
void merge(keytype *left, int sizeLeft, keytype *right, int sizeRight, keytype *temp);
void pmerge(keytype *left, int sizeLeft, keytype *right, int sizeRight, keytype *temp);
int binarySearch(keytype *a, int size, keytype key);
void parallelSort (int N, keytype* A);

void parallelSort (int N, keytype* A)
{
keytype *B = new keytype[N];
#pragma omp parallel
#pragma omp single
mergeSort(A, N, B);
}

void mergeSort(keytype *A, int size, keytype *temp)
{
if(size<=THRESHOLD)
{
sequentialSort(size, A);
return;
}
else
{
int halfSize = size/2;
#pragma omp task
mergeSort(A, halfSize, temp);
mergeSort(A + halfSize, size-halfSize, temp + halfSize);
#pragma omp taskwait
pmerge(A, halfSize, A + halfSize, size-halfSize, temp);
for(int i = 0; i < size; i++)
{
A[i] = temp[i];
}
}
}

void merge(keytype *left, int sizeLeft, keytype *right, int sizeRight, keytype *temp)
{
int i = 0, j = 0, k = 0;
while(i<sizeLeft&&j<sizeRight)
{
if(left[i] < right[j])
{
temp[k++] = left[i++];
}
else
{
temp[k++] = right[j++];
}
}
while(i<sizeLeft)
{
temp[k++] = left[i++];
}
while(j<sizeRight)
{
temp[k++] = right[j++];
}
}

void pmerge(keytype *left, int sizeLeft, keytype *right, int sizeRight, keytype *temp)
{
if(sizeLeft <= THRESHOLD || sizeRight <= THRESHOLD)
{
merge(left, sizeLeft, right, sizeRight, temp);
}
else
{
int p = sizeLeft/2;
int x = left[p];
int q = binarySearch(right, sizeRight, x);
#pragma omp task
pmerge(left, p, right, q, temp);
pmerge(left+p, sizeLeft-p, right+q, sizeRight-q, temp+p+q);
#pragma omp taskwait
}
}

int binarySearch(keytype *a, int size, keytype key)
{
int low = 0;
int high = size-1;
while(low<=high)
{
int mid = (high+low)/2;
if(a[mid] == key)
{
return mid + 1;
}
else if(a[mid] > key)
{
high = mid - 1;
}
else
{
low = mid + 1;
}
}
return low;
}

