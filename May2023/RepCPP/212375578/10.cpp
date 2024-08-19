#include <cstdio>
#include <omp.h>

#include "lib.h"

int main() {
const int size = 30;
int* a = randomArray(size);
int count = 0;

#pragma omp parallel for
for (int i = 0; i < size; i++) {
if (a[i] % 9 == 0) {
#pragma omp atomic
count += 1;
}
}

printArray(a, size);
printf("divisible by 9 count = %d\n", count);
}
