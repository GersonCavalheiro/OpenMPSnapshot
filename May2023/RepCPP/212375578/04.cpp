#include <cstdio>
#include <omp.h>

#include "lib.h"

int minimum(int array[], int size) {
int min = array[0];
for (int i = 0; i < size; i++) {
if (array[i] < min) {
min = array[i];
}
}
return min;
}

int maximum(int array[], int size) {
int max = array[0];
for (int i = 0; i < size; i++) {
if (array[i] > max) {
max = array[i];
}
}
return max;
}

int main() {
int* a = randomArray(10);
int* b = randomArray(10);

omp_set_num_threads(2);
int min;
int max;
#pragma omp parallel
{
#pragma omp if (omp_get_thread_num() == 0)
{
min = minimum(a, 10);
}

#pragma omp if (omp_get_thread_num() == 1)
{
max = maximum(b, 10);
}
}
printf("min of a = %d, max of b = %d\n", min, max);
}
