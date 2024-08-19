
#ifndef CODE_PERFORMANCE_OPTIMIZATION_SEARCH_BEST_H
#define CODE_PERFORMANCE_OPTIMIZATION_SEARCH_BEST_H

#include <assert.h>
#include <cmath>
#include <float.h>
#include <climits>
#include "cosine_similarity.h"


template<typename T>
int SearchBest(const T *const pVecA,  
const int lenA,        
const T *const pVecDB, 
const int lenDB)       
{
assert(lenDB % lenA == 0);
const int featsize = lenA;
const int facenum = lenDB / lenA;

int best_index = -INT_MAX;
T best_similarity = -FLT_MAX;
#pragma omp parallel for num_threads(4)
for (int i = 0; i < facenum; i++) {
T similarity = Cosine_similarity(pVecA, pVecDB + i * featsize, featsize);

if (similarity > best_similarity) {
best_similarity = similarity;
best_index = i;
}
}

return best_index;
}

#endif 
