#include <cstdio>
#include <omp.h>

#include "lib.h"

int getOptimalChunkSize(int arraySize, int threadsCount) {
int chunk = arraySize / threadsCount;
if (chunk * threadsCount < arraySize) {
chunk += 1;
}
return chunk;
}

int main() {
int arraySize = 12;
int a[arraySize];
int b[arraySize];
int c[arraySize];

omp_set_num_threads(3);
#pragma omp parallel for schedule(static, 4)
for (int i = 0; i < arraySize; i++) {
printf("\t th_cnt: %d, th_id: %d, iter: %d\n",
omp_get_num_threads(),
omp_get_thread_num(),
i);
a[i] = omp_get_thread_num();
b[i] = 4 - omp_get_thread_num();
}
printArray(a, arraySize);
printArray(b, arraySize);

omp_set_num_threads(4);
#pragma omp parallel for schedule(dynamic, getOptimalChunkSize(arraySize, omp_get_num_threads()))
for (int i = 0; i < arraySize; i++) {
printf("\t th_cnt: %d, th_id: %d, iter: %d\n",
omp_get_num_threads(),
omp_get_thread_num(),
i);
c[i] = a[i] + b[i];
}
printArray(c, arraySize);
}
