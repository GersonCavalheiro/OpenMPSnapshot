#ifndef OMP_SORT_BOE_MERGE
#define OMP_SORT_BOE_MERGE

#include "omp_sorts_headers.h"

namespace omp_par {
template<typename Num>
void batcherOddEvenMergeSort(Num* arr, int N, double& deltaTime) {
double t = omp_get_wtime();

for (int p = 1; p < N; p = 2*p) {
for (int k = p; k > 0; k /= 2) {
#pragma omp parallel for num_threads(P)
for (int j = k % p; j < N - k; j += 2*k) {
for (int i = 0; i < k; ++i) {
if ((i + j) / (2 * p) == (i + j + k) / (2 * p))
if (arr[i + j] > arr[i + j + k])
std::swap(arr[i + j], arr[i + j + k]);
}
}
}
}

deltaTime += omp_get_wtime() - t;
}

}

#endif