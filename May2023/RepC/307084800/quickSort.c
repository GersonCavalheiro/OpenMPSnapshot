#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
void printArray(int32_t *a, int n) {
printf("%s", "Your array: ");
for (int i = 0; i < n; i++) {
printf("%d ", a[i]);
}
}
int partition(int arr[], int low, int high) {
int i, j, temp, key;
key = arr[low];
i = low + 1;
j = high;
while (1) {
while (i < high && key >= arr[i])
i++;
while (key < arr[j])
j--;
if (i < j) {
temp = arr[i];
arr[i] = arr[j];
arr[j] = temp;
} else {
temp = arr[low];
arr[low] = arr[j];
arr[j] = temp;
return (j);
}
}
}
void quickSort(int arr[], int low, int high) {
int j;
if (low < high) {
j = partition(arr, low, high);
#pragma omp task firstprivate(arr, low, j)
{
quickSort(arr, low, j - 1);
}
#pragma omp task firstprivate(arr, high, j)
{
quickSort(arr, j + 1, high);
}
}
}
int main() {
int mode = 0, N = 0;
int32_t *a;
printf("Please, print the number, illustrating data source: 1 - console, 2 - file \n");
scanf("%d", &mode);
switch (mode) {
case 1: {
printf("%s", "Number of elements: \n");
scanf("%d", &N);
if (N <= 0 || N > 10000) {
printf("%s", "Invalid number of elements");
return -1;
}
a = (int32_t *) malloc(N * sizeof(int32_t));
for (int i = 0; i < N; i++) {
printf("%s %d\n", "element:", i);
scanf("%d", a + i);
}
printArray(a, N);
break;
}
case 2: {
FILE *fp;
char name[] = "data.txt";
if ((fp = fopen(name, "r")) == NULL) {
printf("Can't open file");
getchar();
return -1;
}
fscanf(fp, "%d", &N);
if (N <= 0 || N > 10000) {
printf("%s", "Invalid number of elements");
return -1;
}
a = (int *) malloc(N * sizeof(int));
for (int i = 0; i < N; i++) {
fscanf(fp, "%d", a + i);
}
fclose(fp);
printArray(a, N);
}
}
int j = partition(a, 0, N - 1); 
#pragma omp parallel
{
#pragma omp single
{
quickSort(a, 0, N - 1);
}
}
printArray(a, N);
free(a);
}