#ifndef OMP_COUNTING_SORT
#define OMP_COUNTING_SORT
#include <omp.h>
#include "../utils/utils_1d_array.h"
#include "../utils/utils_sort.h"
#ifndef OMP_THREADS
#define OMP_THREADS 2
#endif
void omp_counting_sort(int arr[], int n)
{
int max = 0;
for(int i = 0; i < n; i++){
if(max < arr[i]){
max = arr[i];
}
}
int count[max+1];
int output[n];
#pragma omp parallel for schedule(static)
for (int i = 0; i <= max; i++) {
count[i] = 0;
}
#pragma omp parallel for schedule(static) reduction(+:count[:])
for (int i = 0; i < n; i++) {
count[arr[i]]++;
}
for (int i = 1; i <= max; i++) {
count[i] += count[i - 1];
}
for (int i = n - 1; i >= 0; i--) {
output[count[arr[i]]- 1] = arr[i];
count[arr[i]]--;
}
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) {
arr[i] = output[i];
}
}
#endif