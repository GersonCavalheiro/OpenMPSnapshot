


#ifndef HIPCPU_RUNTIME_H
#define HIPCPU_RUNTIME_H

#define __HIPCPU__

#ifndef __global__
#define __global__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __constant__
#define __constant__ const
#endif

#ifndef __shared__
#define __shared__ static
#endif


#include <cstddef>
#include <climits>
#include <cstring>
#include <limits>
#include <memory>
#include <cmath>
#include <stdexcept>

#include "detail/runtime.hpp"


using hipcpu::dim3;

#define HIP_KERNEL_NAME(...) __VA_ARGS__

typedef int hipLaunchParm;

#define _hipcpu_runtime (hipcpu::runtime::get())

#define hipLaunchKernelGGL(f, grid, block, shared_mem, stream, ...) \
_hipcpu_runtime.submit_kernel(grid, block, shared_mem, stream, \
[=](){ \
f(__VA_ARGS__); \
})

#define hipLaunchKernel(f, grid, block, shared_mem, stream, ...) \
hipLaunchKernelGGL(f, grid, block, shared_mem, stream, 0, __VA_ARGS__)

#define hipLaunchTask(f, stream, ...) \
_hipcpu_runtime.submit_operation([=](){\
f(__VA_ARGS__); \
}, stream)

#define hipLaunchSequentialKernel(f, stream, scratch_mem, ...) \
_hipcpu_runtime.submit_unparallelized_kernel(scratch_mem, stream, \
[=](){ \
f(__VA_ARGS__); \
})

#define hipLaunchKernelNoBarrier(f, grid, block, stream, ...)


#define HIP_DYNAMIC_SHARED_MEMORY _hipcpu_runtime.dev().get_dynamic_shared_memory()

#define hipThreadIdx_x (_hipcpu_runtime.dev().get_block().get_thread_id().x)
#define hipThreadIdx_y (_hipcpu_runtime.dev().get_block().get_thread_id().y)
#define hipThreadIdx_z (_hipcpu_runtime.dev().get_block().get_thread_id().z)

#define hipBlockIdx_x (_hipcpu_runtime.dev().get_grid().get_block_id().x)
#define hipBlockIdx_y (_hipcpu_runtime.dev().get_grid().get_block_id().y)
#define hipBlockIdx_z (_hipcpu_runtime.dev().get_grid().get_block_id().z)

#define hipBlockDim_x (_hipcpu_runtime.dev().get_block().get_block_dim().x)
#define hipBlockDim_y (_hipcpu_runtime.dev().get_block().get_block_dim().y)
#define hipBlockDim_z (_hipcpu_runtime.dev().get_block().get_block_dim().z)

#define hipGridDim_x (_hipcpu_runtime.dev().get_grid().get_grid_dim().x)
#define hipGridDim_y (_hipcpu_runtime.dev().get_grid().get_grid_dim().y)
#define hipGridDim_z (_hipcpu_runtime.dev().get_grid().get_grid_dim().z)

#define HIP_SYMBOL(X) X

typedef enum hipMemcpyKind {
hipMemcpyHostToHost,
hipMemcpyHostToDevice,
hipMemcpyDeviceToHost,
hipMemcpyDeviceToDevice,
hipMemcpyDefault
} hipMemcpyKind;



#define hipEventDefault hipEvent_t()
#define hipEventBlockingSync 0
#define hipEventDisableTiming 0
#define hipEventInterprocess 0
#define hipEventReleaseToDevice 0
#define hipEventReleaseToSystem 0


#define hipHostMallocDefault 0x0
#define hipHostMallocPortable 0x1
#define hipHostMallocMapped 0x2
#define hipHostMallocWriteCombined 0x4
#define hipHostMallocCoherent 0x40000000
#define hipHostMallocNonCoherent 0x80000000

#define hipHostRegisterPortable 0
#define hipHostRegisterMapped 0


typedef int hipEvent_t;
typedef int hipStream_t;
typedef int hipIpcEventHandle_t;
typedef int hipIpcMemHandle_t;
typedef int hipLimit_t;
typedef int hipFuncCache_t;
typedef int hipCtx_t;
typedef int hipSharedMemConfig;
typedef int hipFuncCache;
typedef int hipJitOption;
typedef int hipDevice_t;
typedef int hipModule_t;
typedef int hipFunction_t;
typedef void* hipDeviceptr_t;
typedef int hipArray;
typedef int* hipArray_const_t;
typedef int hipFuncAttributes;
typedef int hipCtx_t;

typedef int hipTextureObject_t;
typedef int hipSurfaceObject_t;
typedef int hipResourceDesc;
typedef int hipTextureDesc;
typedef int hipResourceViewDesc;
typedef int textureReference;

enum hipError_t
{
hipSuccess,
hipErrorInvalidContext,
hipErrorInvalidKernelFile,
hipErrorMemoryAllocation,
hipErrorInitializationError,
hipErrorLaunchFailure,
hipErrorLaunchOutOfResources,
hipErrorInvalidDevice,
hipErrorInvalidValue,
hipErrorInvalidDevicePointer,
hipErrorInvalidMemcpyDirection,
hipErrorUnknown,
hipErrorInvalidResourceHandle,
hipErrorNotReady,
hipErrorNoDevice,
hipErrorPeerAccessAlreadyEnabled,
hipErrorPeerAccessNotEnabled,
hipErrorRuntimeMemory,
hipErrorRuntimeOther,
hipErrorHostMemoryAlreadyRegistered,
hipErrorHostMemoryNotRegistered,
hipErrorMapBufferObjectFailed,
hipErrorTbd
};

typedef void* hipPitchedPtr;


struct hipDeviceArch_t
{
unsigned hasGlobalInt32Atomics    : 1;
unsigned hasGlobalFloatAtomicExch : 1;
unsigned hasSharedInt32Atomics    : 1;
unsigned hasSharedFloatAtomicExch : 1;
unsigned hasFloatAtomicAdd        : 1;

unsigned hasGlobalInt64Atomics    : 1;
unsigned hasSharedInt64Atomics    : 1;

unsigned hasDoubles               : 1;

unsigned hasWarpVote              : 1;
unsigned hasWarpBallot            : 1;
unsigned hasWarpShuffle           : 1;
unsigned hasFunnelShift           : 1;

unsigned hasThreadFenceSystem     : 1;
unsigned hasSyncThreadsExt        : 1;

unsigned hasSurfaceFuncs          : 1;
unsigned has3dGrid                : 1;
unsigned hasDynamicParallelism    : 1;
};

struct hipDeviceProp_t
{
char name[256];
size_t totalGlobalMem;
size_t sharedMemPerBlock;
int regsPerBlock;
int warpSize;
int maxThreadsPerBlock;
int maxThreadsDim[3];
int maxGridSize[3];
int clockRate;
int memoryClockRate;
int memoryBusWidth;
size_t totalConstMem;
int major;
int minor;
int multiProcessorCount;
int l2CacheSize;
int maxThreadsPerMultiProcessor;
int computeMode;
int clockInstructionRate;
hipDeviceArch_t arch;
int concurrentKernels;
int pciBusID;
int pciDeviceID;
size_t maxSharedMemoryPerMultiProcessor;
int isMultiGpuBoard;
int canMapHostMemory;
int gcnArch;
};


struct hipMemcpy3DParms {};
enum hipDeviceAttribute_t
{
hipDeviceAttributeMaxThreadsPerBlock,
hipDeviceAttributeMaxBlockDimX,
hipDeviceAttributeMaxBlockDimY,
hipDeviceAttributeMaxBlockDimZ,
hipDeviceAttributeMaxGridDimX,
hipDeviceAttributeMaxGridDimY,
hipDeviceAttributeMaxGridDimZ,
hipDeviceAttributeMaxSharedMemoryPerBlock,
hipDeviceAttributeTotalConstantMemory,
hipDeviceAttributeWarpSize,
hipDeviceAttributeMaxRegistersPerBlock,
hipDeviceAttributeClockRate,
hipDeviceAttributeMemoryClockRate,
hipDeviceAttributeMemoryBusWidth,
hipDeviceAttributeMultiprocessorCount,
hipDeviceAttributeComputeMode,
hipDeviceAttributeL2CacheSize,
hipDeviceAttributeMaxThreadsPerMultiProcessor,
hipDeviceAttributeComputeCapabilityMajor,
hipDeviceAttributeComputeCapabilityMinor,
hipDeviceAttributeConcurrentKernels,
hipDeviceAttributePciBusId,
hipDeviceAttributePciDeviceId,
hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
hipDeviceAttributeIsMultiGpuBoard,
hipDeviceAttributeIntegrated,
};

struct hipPointerAttribute_t
{
hipDevice_t device;
hipDeviceptr_t devicePointer;
void* hostPointer;
bool isManaged;
int allocationFlags;
};

#define hipStreamDefault 0
#define hipStreamNonBlocking 0

#define hipSharedMemBankSizeDefault 0
#define hipSharedMemBankSizeFourByte 0
#define hipSharedMemBankSizeEightByte 0

typedef void(*hipStreamCallback_t)(hipStream_t, hipError_t, void*);




inline
hipError_t hipMalloc(void** ptr, size_t size)
{
*ptr = hipcpu::detail::aligned_malloc(hipcpu::detail::default_alignment, size);

if(*ptr == nullptr)
return hipErrorMemoryAllocation;

return hipSuccess;
}


inline
hipError_t hipFree(void* ptr)
{
hipcpu::detail::aligned_free(ptr);
return hipSuccess;
}

inline
hipError_t hipMallocHost(void** ptr, size_t size)
{
return hipMalloc(ptr, size);
}

#define hipMemAttachGlobal 0
#define hipMemAttachHost 1

template<class T>
inline
hipError_t hipMallocManaged(T** ptr, size_t size, unsigned flags = hipMemAttachGlobal)
{
return hipMalloc(reinterpret_cast<void**>(ptr), size);
}

inline
hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags)
{
return hipMalloc(ptr, size);
}

inline
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags)
{
return hipMalloc(ptr, size);
}



inline
hipError_t hipFreeHost(void* ptr)
{
return hipFree(ptr);
}

inline
hipError_t hipHostFree(void* ptr)
{
return hipFree(ptr);
}

inline
hipError_t hipSetDevice(int device)
{
if(device != 0)
return hipErrorInvalidDevice;

_hipcpu_runtime.set_device(device);
return hipSuccess;
}

inline
hipError_t hipStreamCreate(hipStream_t* stream)
{
*stream = _hipcpu_runtime.create_blocking_stream();
return hipSuccess;
}

inline
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags)
{
if(flags == hipStreamDefault)
return hipStreamCreate(stream);
else if (flags == hipStreamNonBlocking) 
{
*stream = _hipcpu_runtime.create_async_stream();
return hipSuccess;
}

return hipErrorInvalidValue;
}

inline
hipError_t hipStreamSynchronize(hipStream_t stream)
{
_hipcpu_runtime.streams().get(stream)->wait();
return hipSuccess;
}

inline
hipError_t hipStreamDestroy(hipStream_t stream)
{
_hipcpu_runtime.destroy_stream(stream);
return hipSuccess;
}

inline
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
unsigned int flags)
{
std::shared_ptr<hipcpu::event> evt = _hipcpu_runtime.events().get_shared(event);
_hipcpu_runtime.submit_operation([evt](){
evt->wait();
}, stream);
return hipSuccess;
}

inline
hipError_t hipStreamQuery(hipStream_t stream)
{
hipcpu::stream* s = _hipcpu_runtime.streams().get(stream);

if(s->is_idle())
return hipSuccess;

return hipErrorNotReady;
}

inline
hipError_t hipStreamAddCallback(hipStream_t stream,
hipStreamCallback_t callback, void *userData,
unsigned int flags) 
{
_hipcpu_runtime.submit_operation([stream, callback, userData](){
callback(stream, hipSuccess, userData);
}, stream);
return hipSuccess;
}

inline
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
hipMemcpyKind copyKind, hipStream_t stream = 0)
{
if(!_hipcpu_runtime.streams().is_valid(stream))
return hipErrorInvalidValue;

_hipcpu_runtime.submit_operation([=](){
memcpy(dst, src, sizeBytes);
}, stream);

return hipSuccess;
}

inline                                            
hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
hipMemcpyKind copyKind)
{
hipMemcpyAsync(dst, src, sizeBytes, copyKind, 0);
_hipcpu_runtime.streams().get(0)->wait();
return hipSuccess;
}

inline
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t size)
{
return hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}

inline
hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t size)
{
return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
}

inline
hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size)
{
return hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
}

inline
hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t size,
hipStream_t stream)
{
return hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream);
}

inline
hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t size,
hipStream_t stream)
{
return hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream);
}

inline
hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size,
hipStream_t stream)
{
return hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);
}

inline
hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src,
size_t sizeBytes, size_t offset,
hipMemcpyKind copyType,
hipStream_t stream = 0)
{
char* base_ptr = static_cast<char*>(const_cast<void*>(symbol));
void* ptr = static_cast<void*>(base_ptr + offset);
return hipMemcpyAsync(ptr, src, sizeBytes, copyType, stream);
}

inline
hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName,
size_t sizeBytes, size_t offset,
hipMemcpyKind kind,
hipStream_t stream = 0)
{
const void* ptr = 
static_cast<const void*>(static_cast<const char*>(symbolName)+offset);
return hipMemcpyAsync(dst, ptr, sizeBytes, kind, stream);
}

inline
hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes,
size_t offset = 0,
hipMemcpyKind copyType = hipMemcpyHostToDevice)
{
hipError_t err = 
hipMemcpyToSymbolAsync(symbol, src, sizeBytes, offset, copyType, 0);

if(err != hipSuccess)
return err;

_hipcpu_runtime.streams().get(0)->wait();
return err;
}

inline
hipError_t hipMemcpyFromSymbol(void *dst, const void *symbolName,
size_t sizeBytes, size_t offset = 0,
hipMemcpyKind kind = hipMemcpyDeviceToHost) 
{
hipError_t err = 
hipMemcpyFromSymbolAsync(dst, symbolName, sizeBytes, offset, kind, 0);

if(err != hipSuccess)
return err;

_hipcpu_runtime.streams().get(0)->wait();
return err;
}


hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p);

inline
hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
size_t width, size_t height, hipMemcpyKind kind,
hipStream_t stream)
{
if(!_hipcpu_runtime.streams().is_valid(stream))
return hipErrorInvalidValue;

_hipcpu_runtime.submit_operation([=](){
for(size_t row = 0; row < height; ++row)
{
void* row_dst_begin = reinterpret_cast<char*>(dst) + row * dpitch;
const void* row_src_begin = reinterpret_cast<const char*>(src) + row * spitch;

memcpy(row_dst_begin, row_src_begin, width);
}
}, stream);

return hipSuccess;
}

inline
hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
size_t width, size_t height, hipMemcpyKind kind)
{
hipError_t err = hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, 0);

if(err != hipSuccess)
return err;

_hipcpu_runtime.streams().get(0)->wait();
return err;
}

hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset,
const void* src, size_t spitch, size_t width,
size_t height, hipMemcpyKind kind);

hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset,
const void* src, size_t count, hipMemcpyKind kind);

hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset,
size_t hOffset, size_t count, hipMemcpyKind kind);

hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset,
size_t count);

hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost,
size_t count);

inline
hipError_t hipDeviceSynchronize()
{
_hipcpu_runtime.streams().for_each([](hipcpu::stream* s){
s->wait();
});
return hipSuccess;
}

hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* pCacheConfig);

const char* hipGetErrorString(hipError_t error);

const char* hipGetErrorName(hipError_t error);

inline
hipError_t hipGetDeviceCount(int* count)
{
*count = 1;
return hipSuccess;
}

inline
hipError_t hipGetDevice(int* device)
{
*device = 0;
return hipSuccess;
}


inline
hipError_t hipMemsetAsync(void* devPtr, int value, size_t count,
hipStream_t stream = 0)
{
if(!_hipcpu_runtime.streams().is_valid(stream))
return hipErrorInvalidValue;

_hipcpu_runtime.submit_operation([=](){
memset(devPtr, value, count);
}, stream);

return hipSuccess;
}

inline
hipError_t hipMemset(void* devPtr, int value, size_t count)
{
hipError_t err = hipMemsetAsync(devPtr, value, count, 0);
if(err != hipSuccess)
return err;

_hipcpu_runtime.streams().get(0)->wait();
return hipSuccess;
}

inline
hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes)
{
return hipMemset(dest, value, sizeBytes);
}



inline
hipError_t hipGetDeviceProperties(hipDeviceProp_t* p_prop, int device)
{
if(device != 0)
return hipErrorInvalidDevice;

static const char device_name[] = "hipCPU OpenMP host device";
int max_dim = std::numeric_limits<int>::max();

static_assert(sizeof device_name <= sizeof p_prop->name);
memcpy(p_prop->name, device_name, sizeof device_name);

p_prop->totalGlobalMem = std::numeric_limits<size_t>::max();
p_prop->sharedMemPerBlock = _hipcpu_runtime.dev().get_max_shared_memory();
p_prop->regsPerBlock = std::numeric_limits<int>::max();
p_prop->warpSize = 1;
p_prop->maxThreadsPerBlock = _hipcpu_runtime.dev().get_max_threads();
p_prop->maxGridSize[0] = max_dim;
p_prop->maxGridSize[1] = max_dim;
p_prop->maxGridSize[2] = max_dim;
p_prop->maxGridSize[0] = max_dim;
p_prop->maxGridSize[1] = max_dim;
p_prop->maxGridSize[2] = max_dim;
p_prop->clockRate = 1;
p_prop->memoryClockRate = 1;
p_prop->memoryBusWidth = 1;
p_prop->totalConstMem = std::numeric_limits<std::size_t>::max();
p_prop->major = 1;
p_prop->minor = 0;
p_prop->multiProcessorCount = _hipcpu_runtime.dev().get_num_compute_units();
p_prop->l2CacheSize = std::numeric_limits<int>::max();
p_prop->maxThreadsPerMultiProcessor = p_prop->maxThreadsPerBlock;
p_prop->computeMode = 0;
p_prop->clockInstructionRate = p_prop->clockRate;

hipDeviceArch_t arch;
arch.hasGlobalInt32Atomics = 1;
arch.hasGlobalFloatAtomicExch = 1;
arch.hasSharedInt32Atomics = 1;
arch.hasSharedFloatAtomicExch = 1;
arch.hasFloatAtomicAdd = 1;
arch.hasGlobalInt64Atomics = 1;
arch.hasSharedInt64Atomics = 1;
arch.hasDoubles = 1;
arch.hasWarpVote = 0;
arch.hasWarpBallot = 0;
arch.hasWarpShuffle = 0;
arch.hasFunnelShift = 0;
arch.hasThreadFenceSystem = 1;
arch.hasSyncThreadsExt = 1;
arch.hasSurfaceFuncs = 0;
arch.has3dGrid = 1;
arch.hasDynamicParallelism = 0;

p_prop->arch = arch;
p_prop->concurrentKernels = 1;
p_prop->pciBusID = 0;
p_prop->pciDeviceID = 0;
p_prop->maxSharedMemoryPerMultiProcessor = p_prop->sharedMemPerBlock;
p_prop->isMultiGpuBoard = 0;
p_prop->canMapHostMemory = 1;
p_prop->gcnArch = 0;

return hipSuccess;
}

hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device);

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
const void* func,
int blockSize,
size_t dynamicSMemSize);

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, void* ptr);

hipError_t hipMemGetInfo(size_t* free, size_t* total);

inline
hipError_t hipEventCreate(hipEvent_t* event)
{
*event = _hipcpu_runtime.create_event();
return hipSuccess;
}

inline
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = 0)
{
if(!_hipcpu_runtime.events().is_valid(event) ||
!_hipcpu_runtime.streams().is_valid(stream))
return hipErrorInvalidValue;

std::shared_ptr<hipcpu::event> evt = _hipcpu_runtime.events().get_shared(event);
_hipcpu_runtime.submit_operation([evt](){
evt->mark_as_finished();
}, stream);
return hipSuccess;
}

inline
hipError_t hipEventSynchronize(hipEvent_t event)
{
if(!_hipcpu_runtime.events().is_valid(event))
return hipErrorInvalidValue;

hipcpu::event* evt = _hipcpu_runtime.events().get(event);
evt->wait();

if(evt->is_complete())
return hipSuccess;

return hipErrorUnknown;
}

inline
hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop)
{
if(!_hipcpu_runtime.events().is_valid(start) || !_hipcpu_runtime.events().is_valid(stop))
return hipErrorInvalidValue;

hipcpu::event* start_evt = _hipcpu_runtime.events().get(start);
hipcpu::event* stop_evt = _hipcpu_runtime.events().get(stop);
if(start_evt->is_complete() && stop_evt->is_complete())
{
*ms = static_cast<float>(stop_evt->timestamp_ns() - start_evt->timestamp_ns()) / 1e6f;
return hipSuccess;
}

return hipErrorUnknown;
}

inline
hipError_t hipEventDestroy(hipEvent_t event)
{
if(!_hipcpu_runtime.events().is_valid(event))
return hipErrorInvalidValue;

_hipcpu_runtime.destroy_event(event);
return hipSuccess;
}

hipError_t hipDriverGetVersion(int* driverVersion);

inline
hipError_t hipRuntimeGetVersion(int* runtimeVersion)
{
*runtimeVersion = 99999;
return hipSuccess;
}

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);

hipError_t hipDeviceDisablePeerAccess(int peerDevice);

hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);

hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags,
int* active);

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
hipDeviceptr_t dptr);

hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
size_t count);

hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
int srcDevice, size_t count,
hipStream_t stream = 0);

hipError_t hipProfilerStart();
hipError_t hipProfilerStop();

hipError_t hipSetDeviceFlags(unsigned int flags);

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);

inline
hipError_t hipEventQuery(hipEvent_t event)
{
if(!_hipcpu_runtime.events().is_valid(event))
return hipErrorInvalidValue;

bool is_ready = _hipcpu_runtime.events().get(event)->is_complete();

if(!is_ready)
return hipErrorNotReady;
return hipSuccess;
}



template <class T>
hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
size_t dynamicSMemSize = 0,
int blockSizeLimit = 0,
unsigned int flags = 0);



#define HIPCPU_MAKE_VECTOR1(T, name) \
struct name {\
T x; \
};


#define HIPCPU_MAKE_VECTOR2(T, name) \
struct name {\
T x; \
T y; \
};


#define HIPCPU_MAKE_VECTOR3(T, name) \
struct name {\
T x; \
T y; \
T z; \
};


#define HIPCPU_MAKE_VECTOR4(T, name) \
struct name {\
T x; \
T y; \
T z; \
T w; \
};

#define HIPCPU_MAKE_VECTOR_TYPE(T, prefix) \
HIPCPU_MAKE_VECTOR1(T, prefix##1) \
HIPCPU_MAKE_VECTOR2(T, prefix##2) \
HIPCPU_MAKE_VECTOR3(T, prefix##3) \
HIPCPU_MAKE_VECTOR4(T, prefix##4)


HIPCPU_MAKE_VECTOR_TYPE(signed char, char)
HIPCPU_MAKE_VECTOR_TYPE(unsigned char, uchar)
HIPCPU_MAKE_VECTOR_TYPE(short, short)
HIPCPU_MAKE_VECTOR_TYPE(unsigned short, ushort)
HIPCPU_MAKE_VECTOR_TYPE(int, int)
HIPCPU_MAKE_VECTOR_TYPE(unsigned, uint)
HIPCPU_MAKE_VECTOR_TYPE(long, long)
HIPCPU_MAKE_VECTOR_TYPE(unsigned long, ulong)
HIPCPU_MAKE_VECTOR_TYPE(long long, longlong)
HIPCPU_MAKE_VECTOR_TYPE(unsigned long long, ulonglong)
HIPCPU_MAKE_VECTOR_TYPE(float, float)
HIPCPU_MAKE_VECTOR_TYPE(double, double)



__device__
inline
void __syncthreads()
{
#pragma omp barrier
}

__device__
inline
float __fadd_rd(float x, float y)
{
return x+y;
}

__device__
inline
float __fadd_rn(float x, float y)
{
return x+y;
}

__device__
inline
float __fadd_ru(float x, float y)
{
return x+y;
}

__device__
inline
float __fadd_rz(float x, float y)
{
return x+y;
}

__device__
inline
float __fdiv_rd(float x, float y)
{
return x/y;
}

__device__
inline
float __fdiv_rn(float x, float y)
{
return x/y;
}

__device__
inline
float __fdiv_ru(float x, float y)
{
return x/y;
}

__device__
inline
float __fdiv_rz(float x, float y)
{
return x/y;
}

__device__
inline
float __fdividef(float x, float y)
{
return x/y;
}

__device__
inline
float __fmaf_rd(float x, float y, float z)
{
return std::fma(x,y,z);
}

__device__
inline
float __fmaf_rn(float x, float y, float z)
{
return std::fma(x,y,z);
}

__device__
inline
float __fmaf_ru(float x, float y, float z)
{
return std::fma(x,y,z);
}

__device__
inline
float __fmaf_rz(float x, float y, float z)
{
return std::fma(x,y,z);
}

__device__
inline
float __fmul_rd(float x, float y)
{
return x*y;
}

__device__
inline
float __fmul_rn(float x, float y)
{
return x*y;
}

__device__
inline
float __fmul_ru(float x, float y)
{
return x*y;
}

__device__
inline
float __fmul_rz(float x, float y)
{
return x*y;
}

__device__
inline
float __frcp_rd(float x)
{
return 1.f/x;
}

__device__
inline
float __frcp_rn(float x)
{
return 1.f/x;
}

__device__
inline
float __frcp_ru(float x)
{
return 1.f/x;
}

__device__
inline
float __frcp_rz(float x)
{
return 1.f/x;
}

__device__
inline
float __frsqrt_rn(float x)
{
return 1.f/std::sqrt(x);
}

__device__
inline
float __fsqrt_rd(float x)
{
return std::sqrt(x);
}

__device__
inline
float __fsqrt_rn(float x)
{
return std::sqrt(x);
}

__device__
inline
float __fsqrt_ru(float x)
{
return std::sqrt(x);
}

__device__
inline
float __fsqrt_rz(float x)
{
return std::sqrt(x);
}

__device__
inline
float __fsub_rd(float x, float y)
{
return x-y;
}

__device__
inline
float __fsub_rn(float x, float y)
{
return x-y;
}

__device__
inline
float __fsub_ru(float x, float y)
{
return x-y;
}

__device__
inline
float __fsub_rz(float x, float y)
{
return x-y;
}

__device__
inline
double __dadd_rd(double x, double y)
{
return x+y;
}

__device__
inline
double __dadd_rn(double x, double y)
{
return x+y;
}

__device__
inline
double __dadd_ru(double x, double y)
{
return x+y;
}

__device__
inline
double __dadd_rz(double x, double y)
{
return x+y;
}

__device__
inline
double __ddiv_rd(double x, double y)
{
return x/y;
}

__device__
inline
double __ddiv_rn(double x, double y)
{
return x/y;
}

__device__
inline
double __ddiv_ru(double x, double y)
{
return x/y;
}

__device__
inline
double __ddiv_rz(double x, double y)
{
return x/y;
}

__device__
inline
double __dmul_rd(double x, double y)
{
return x*y;
}

__device__
inline
double __dmul_rn(double x, double y)
{
return x*y;
}

__device__
inline
double __dmul_ru(double x, double y)
{
return x*y;
}

__device__
inline
double __dmul_rz(double x, double y)
{
return x*y;
}

__device__
inline
double __drcp_rd(double x)
{
return 1./x;
}

__device__
inline
double __drcp_rn(double x)
{
return 1./x;
}

__device__
inline
double __drcp_ru(double x)
{
return 1./x;
}

__device__
inline
double __drcp_rz(double x)
{
return 1./x;
}

__device__
inline
double __dsqrt_rd(double x)
{
return std::sqrt(x);
}

__device__
inline
double __dsqrt_rn(double x)
{
return std::sqrt(x);
}

__device__
inline
double __dsqrt_ru(double x)
{
return std::sqrt(x);
}

__device__
inline
double __dsqrt_rz(double x)
{
return std::sqrt(x);
}

__device__
inline
double __dsub_rd(double x, double y)
{
return x - y;
}

__device__
inline
double __dsub_rn(double x, double y)
{
return x - y;
}

__device__
inline
double __dsub_ru(double x, double y)
{
return x - y;
}

__device__
inline
double __dsub_rz(double x, double y)
{
return x - y;
}

__device__
inline
double __fma_rd(double x, double y, double z)
{
return std::fma(x,y,z);
}

__device__
inline
double __fma_rn(double x, double y, double z)
{
return std::fma(x,y,z);
}

__device__
inline
double __fma_ru(double x, double y, double z)
{
return std::fma(x,y,z);
}

__device__
inline
double __fma_rz(double x, double y, double z)
{
return std::fma(x,y,z);
}


#endif 
