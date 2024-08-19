#pragma once
#include <cub/cub.cuh>
#include "cutil_subset.h"

class GlobalBarrier {
public:
typedef unsigned int SyncFlag;

protected :
SyncFlag* d_sync;

__device__ __forceinline__ SyncFlag LoadCG(SyncFlag* d_ptr) const {
SyncFlag retval;
retval = cub::ThreadLoad<cub::LOAD_CG>(d_ptr);
return retval;
}

public:
GlobalBarrier() : d_sync(NULL) {}

__device__ __forceinline__ void Sync() const {
volatile SyncFlag *d_vol_sync = d_sync;

__threadfence();
__syncthreads();

if (blockIdx.x == 0) {
if (threadIdx.x == 0) {
d_vol_sync[blockIdx.x] = 1;
}
__syncthreads();

for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
while (LoadCG(d_sync + peer_block) == 0) {
__threadfence_block();
}
}
__syncthreads();

for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
d_vol_sync[peer_block] = 0;
}
} else {
if (threadIdx.x == 0) {
d_vol_sync[blockIdx.x] = 1;

while (LoadCG(d_sync + blockIdx.x) == 1) {
__threadfence_block();
}
}
__syncthreads();
}
}
};


class GlobalBarrierLifetime : public GlobalBarrier {
protected:
size_t sync_bytes;

public:
GlobalBarrierLifetime() : GlobalBarrier(), sync_bytes(0) {}

cudaError_t HostReset() {
cudaError_t retval = cudaSuccess;
if (d_sync) {
CUDA_SAFE_CALL(cudaFree(d_sync));
d_sync = NULL;
}
sync_bytes = 0;
return retval;
}

virtual ~GlobalBarrierLifetime() {
HostReset();
}

cudaError_t Setup(int sweep_grid_size) {
cudaError_t retval = cudaSuccess;
do {
size_t new_sync_bytes = sweep_grid_size * sizeof(SyncFlag);
if (new_sync_bytes > sync_bytes) {
if (d_sync) {
CUDA_SAFE_CALL(cudaFree(d_sync));
retval = cudaSuccess;
}
sync_bytes = new_sync_bytes;
CUDA_SAFE_CALL(cudaMalloc((void**) &d_sync, sync_bytes));
retval = cudaSuccess;
CUDA_SAFE_CALL(cudaMemset(d_sync, 0, sweep_grid_size * sizeof(SyncFlag)));
}
} while (0);
return retval;
}
};
