#include <cstdio>
#include <omp.h>

#include "lib.h"

int main() {
const int size = 40;
int* array = randomArray(size);
int max = -1;

#pragma omp parallel for
for (int i = 0; i < size; i++) {
if (array[i] % 7 == 0) {
#pragma omp critical
if (array[i] > max) {
max = array[i];
}
}
}

printArray(array, size);
printf("max divisible by 7 = %d\n", max);
}
