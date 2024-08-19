#include <iostream>
#include <omp.h>
#include "Mergesort.h"
using namespace std;
void Mergesort::merge(int32_t* arr, int l, int m, int r)
{
int i, j, k;
int n1 = m - l + 1;
int n2 = r - m;
int32_t *L = (int32_t *)malloc(sizeof(int32_t) * n1);
int32_t *R = (int32_t *)malloc(sizeof(int32_t) * n2);
for (i = 0; i < n1; i++)
L[i] = arr[l + i];
for (j = 0; j < n2; j++)
R[j] = arr[m + 1 + j];
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
free(L);
free(R);
}
void Mergesort::sort(int32_t* arr, int size)
{
for (int i = 1; i <= size - 1; i += i)
{
for (int j = 0; j < size - 1; j += 2 * i)
{
int mid = min(j + i - 1, size - 1);
int rightEnd = min(j + 2 * i - 1, size - 1);
merge(arr, j, mid, rightEnd);
}
}
}
void Mergesort::sortParallel(int32_t* arr, int size)
{
cout << "sorting parallel ... " << endl;
int incr = 1;
#pragma omp parallel for
for (int i = incr; i <= size - 1; i += incr)
{
for (int j = 0; j < size - 1; j += i * 2)
{
int mid = min(j + i - 1, size - 1);
int rightEnd = min(j + 2 * i - 1, size - 1);
merge(arr, j, mid, rightEnd);
}
incr = i;
}
}
void Mergesort::fillWithRandomNumbers(int32_t* arr, int size) {
unsigned int seed = time(NULL);
for (int i = 0; i < size; ++i)
{
arr[i] = rand_r(&seed) % size;
}
}
void Mergesort::print(const int32_t* arr, int size)
{
if (size <= 0)
return;
cout << "[" << arr[0];
if (size <= MAX_PRINT_SIZE)
{
for (int i = 1; i < size; ++i)
{
cout << ", " << arr[i];
}
}
else
{
int i;
for (i = 1; i < MAX_PRINT_SIZE / 2; ++i)
{
cout << ", " << arr[i];
}
cout << ", ...";
for (i = size - i; i < size; ++i)
{
cout << ", " << arr[i];
}
}
cout << "]" << endl;
}
