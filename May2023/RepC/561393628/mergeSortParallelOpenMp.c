#include "mergeSortParallelOpenMp.h"
#include <stdio.h>
#include <omp.h>
#include "../mergeSortSequential/mergeSortSequential.h"
#include "../commonFunctions/commonFunctions.h"
void mergeSortParallel(int array[], int size, int bufferArray[]);
void mergeSortParallelOpenMp(int array[], int size, int threadsNumber) {
omp_set_num_threads(threadsNumber);
printf("Number of maximum thread: %d\n", omp_get_max_threads());
printf("Using %d threads\n", omp_get_max_threads());
int* bufferArray = allocateMemory(size * sizeof(int));
#pragma omp parallel
#pragma omp single
mergeSortParallel(array, size, bufferArray);
free(bufferArray);
}
void mergeSortParallel(int array[], int size, int bufferArray[]) {
if (size < MULTITHREAD_THRESHOLD)
{
mergeSort(array, size, bufferArray);
return;
}
#pragma omp task
mergeSortParallel(array, size / 2, bufferArray);
mergeSortParallel(array + size / 2, size - size / 2, bufferArray + size / 2);
#pragma omp taskwait
merge(array, size, bufferArray);
}
