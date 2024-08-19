

#ifndef TAHIP_H
#define TAHIP_H

#include <hipblas.h>
#include <hip/hip_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

typedef void *tahipRequest;

static const tahipRequest TAHIP_REQUEST_NULL = NULL;
static const size_t TAHIP_STREAMS_AUTO = 0;

hipError_t
tahipInit(unsigned int flags);

hipError_t
tahipFinalize();

hipError_t
tahipCreateStreams(size_t count);

hipError_t
tahipDestroyStreams();

hipError_t
tahipGetStream(hipStream_t *stream);

hipError_t
tahipReturnStream(hipStream_t stream);

hipError_t
tahipSynchronizeStreamAsync(hipStream_t stream);



__host__ __device__ hipError_t
tahipMemcpyAsync(
void *dst, const void *src, size_t sizeBytes,
enum hipMemcpyKind kind, hipStream_t stream,
tahipRequest *request);

__host__ __device__ hipError_t
tahipMemsetAsync(
void *devPtr, int value, size_t sizeBytes,
hipStream_t stream,
tahipRequest *request);

__host__ hipError_t
tahipLaunchKernel(
const void* func, dim3 gridDim, dim3 blockDim,
void** args, size_t sharedMem, hipStream_t stream,
tahipRequest *request);

__device__ hipblasStatus_t
tahipblasGemmEx(hipblasHandle_t handle,
hipblasOperation_t transa, hipblasOperation_t transb, int m,
int n, int k, const void *alpha, const void *matA, hipblasDatatype_t Atype,
int lda, const void *matB, hipblasDatatype_t Btype, int ldb,
const void *beta, void *matC, hipblasDatatype_t Ctype, int ldc, hipblasDatatype_t computeType,
hipblasGemmAlgo_t algo, hipStream_t stream, tahipRequest *requestPtr);

hipError_t
tahipWaitRequestAsync(tahipRequest *request);

hipError_t
tahipWaitallRequestsAsync(size_t count, tahipRequest requests[]);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif 
