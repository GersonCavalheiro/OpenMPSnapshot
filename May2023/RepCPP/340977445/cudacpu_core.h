#ifndef CUDACPU_CORE_H
#define CUDACPU_CORE_H


#include "cudacpu_struct.h"
#include "cudacpu_extra.h"

#include <cassert>
#include <omp.h>
#include <utility>

dim3  gridDim;
dim3  blockDim;

int3 blockIdx;
int3 threadIdx;
#pragma omp threadprivate(threadIdx)

constexpr int warpSize = 32;

template <class Func, class ... Params>
void CudaCpuCore(const int3 inIdxBlock, const dim3 inBlockDim, Func&& inFunc,
Params ... params){
const int totalNbThreads = dim3_mul(inBlockDim);
assert(totalNbThreads);

blockIdx = inIdxBlock;

#pragma omp parallel num_threads(totalNbThreads)
{
threadIdx = dim3_1DidxToPos(omp_get_thread_num(), inBlockDim);
inFunc(params...);
}
}

template <class Func, class ... Params>
void CudaCpu(const dim3 inGridSize, const dim3 inBlockDim, Func&& inFunc,
Params ... params){

gridDim = inGridSize;
blockDim = inBlockDim;

for(int idxBlockX = 0 ; idxBlockX < inGridSize.x ; ++idxBlockX){
for(int idxBlockY = 0 ; idxBlockY < inGridSize.y ; ++idxBlockY){
for(int idxBlockZ = 0 ; idxBlockZ < inGridSize.z ; ++idxBlockZ){
CudaCpuCore(make_int3(idxBlockX,idxBlockY,idxBlockZ), inBlockDim, std::forward<Func>(inFunc),
std::forward<Params>(params)...);
}
}
}
}

#endif
