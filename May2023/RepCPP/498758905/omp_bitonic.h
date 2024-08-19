#ifndef OMP_SORT_BITONIC
#define OMP_SORT_BITONIC

#include "omp_sorts_headers.h"

namespace omp_par {
template<typename Num>
void bitonicSort(Num* arr, int N, double& deltaTime) {        
double t = omp_get_wtime();

for (int runSize = 2; runSize <= N; runSize = 2*runSize) {
for (int stage = runSize / 2; stage > 0; stage /= 2) {
#pragma omp parallel for num_threads(P)
for (int wireLowerEnd = 0; wireLowerEnd < N; ++wireLowerEnd) {
int wireUpperEnd = wireLowerEnd ^ stage;
if (wireUpperEnd > wireLowerEnd){
if (((wireLowerEnd & runSize) == 0 && arr[wireLowerEnd] > arr[wireUpperEnd]) 
|| ((wireLowerEnd & runSize) != 0 && arr[wireLowerEnd] < arr[wireUpperEnd])) 
std::swap(arr[wireLowerEnd], arr[wireUpperEnd]);
}
}
}
}

deltaTime += omp_get_wtime() - t;
}

}

#endif