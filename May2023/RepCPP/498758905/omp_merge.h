#ifndef OMP_SORT_MERGE
#define OMP_SORT_MERGE

#include "omp_sorts_headers.h"
#include "../sequential/insertion.h"
#include "../sequential/merge.h"

#include <stdio.h>

namespace omp_par {
template<typename Num>
void mergeSort(Num* arr, int N, double& deltaTime) {    
double t = omp_get_wtime();

register int currentSize = seq::getMinRunSize(N), i;

if (currentSize > 1) {
#pragma omp parallel for num_threads(P)
for (i = 0; i < N; i += currentSize)
seq::insertionSort(arr + i, i + currentSize < N ? currentSize : N - i);
}
for (; currentSize < N; currentSize *= 2) {
#pragma omp parallel for num_threads(P)
for (i = 0; i < N - 1; i += 2*currentSize) {
int mid = std::min(i + currentSize, N-1);
int right = std::min(i + 2*currentSize - 1, N-1);

if (arr[mid - 1] > arr[mid]) 
seq::merge(arr, i, mid, right);
}
}

deltaTime += omp_get_wtime() - t;
}

}

#endif