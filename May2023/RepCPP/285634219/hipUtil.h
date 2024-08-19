#include <hip/hip_runtime.h>
#pragma once
#include <stdio.h>

#define hipCheckError() {                                                       \
hipError_t e=hipGetLastError();                                                \
if(e!=hipSuccess) {                                                            \
printf("HIP failure %s:%d: '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));  \
exit(0);                                                                     \
}                                                                              \
}

__device__  __forceinline__
void atomicWarpReduceAndUpdate(POSVEL_T *out, POSVEL_T val) {
val+=__shfl_down(val, 16, 32); 
val+=__shfl_down(val, 8, 32); 
val+=__shfl_down(val, 4, 32);
val+=__shfl_down(val, 2, 32); 
val+=__shfl_down(val, 1, 32);

if(hipThreadIdx_x%32==0)
atomicAdd(out,val);  
}

class hipDeviceSelector {
public:
hipDeviceSelector() {
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

if((str = getenv("HACC_NUM_HIP_DEV")) != NULL) {
numDev=atoi(str);
}

int dev;
dev = local_rank % numDev;

hipSetDevice(dev);
}
};
