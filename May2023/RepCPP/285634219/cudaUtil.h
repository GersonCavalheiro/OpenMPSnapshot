#include <cuda.h>
#pragma once
#include <stdio.h>

#define cudaCheckError() {                                                       \
cudaError_t e=cudaGetLastError();                                                \
if(e!=cudaSuccess) {                                                            \
printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));  \
exit(0);                                                                     \
}                                                                              \
}

__device__  __forceinline__
void atomicWarpReduceAndUpdate(POSVEL_T *out, POSVEL_T val) {
val+=__shfl_down(val, 16); 
val+=__shfl_down(val, 8); 
val+=__shfl_down(val, 4);
val+=__shfl_down(val, 2); 
val+=__shfl_down(val, 1);

if(threadIdx.x%32==0)
atomicAdd(out,val);  
}

class cudaDeviceSelector {
public:
cudaDeviceSelector() {
char* str;
int local_rank = 0;
int numDev=1;

if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
local_rank = atoi(str);
}
else if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
local_rank = atoi(str);
}
else if((str = getenv("SLURM_LOCALID")) != NULL) {
local_rank = atoi(str);
}

if((str = getenv("HACC_NUM_CUDA_DEV")) != NULL) {
numDev=atoi(str);
}

#if 0

#if 0
char var[100];
sprintf(var,"/tmp/nvidia-mps_%d",local_rank%numDev);
setenv("CUDA_MPS_PIPE_DIRECTORY",var,1);
#endif
#else 
int dev;
dev = local_rank % numDev;

cudaSetDevice(dev);
#endif
}
};
