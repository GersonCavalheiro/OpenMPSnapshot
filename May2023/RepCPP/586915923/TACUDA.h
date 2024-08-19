

#ifndef TACUDA_H
#define TACUDA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <library_types.h>

#include <stddef.h>
#include <stdint.h>

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

typedef void *tacudaRequest;

static const tacudaRequest TACUDA_REQUEST_NULL = NULL;
static const size_t TACUDA_STREAMS_AUTO = 0;

CUresult
tacudaInit(unsigned int flags);

CUresult
tacudaFinalize();

cudaError_t
tacudaCreateStreams(size_t count);

cudaError_t
tacudaDestroyStreams();

cudaError_t
tacudaGetStream(cudaStream_t *stream);

cudaError_t
tacudaReturnStream(cudaStream_t stream);

cudaError_t
tacudaSynchronizeStreamAsync(cudaStream_t stream);



__host__ __device__ cudaError_t
tacudaMemcpyAsync(
void *dst, const void *src, size_t sizeBytes,
enum cudaMemcpyKind kind, cudaStream_t stream,
tacudaRequest *request);

__host__ __device__ cudaError_t
tacudaMemsetAsync(
void *devPtr, int value, size_t sizeBytes,
cudaStream_t stream,
tacudaRequest *request);

__host__ cudaError_t
tacudaLaunchKernel(
const void* func, dim3 gridDim, dim3 blockDim, 
void** args, size_t sharedMem, cudaStream_t stream,
tacudaRequest *request);

__device__ cublasStatus_t
tacublasGemmEx(cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb, int m, 
int n, int k, const void *alpha, const void *matA, enum cudaDataType_t Atype,
int lda, const void *matB, enum cudaDataType_t Btype, int ldb,
const void *beta, void *matC, enum cudaDataType_t Ctype, int ldc, enum cudaDataType_t computeType,
cublasGemmAlgo_t algo, cudaStream_t stream, tacudaRequest *requestPtr);

cudaError_t
tacudaWaitRequestAsync(tacudaRequest *request);

cudaError_t
tacudaWaitallRequestsAsync(size_t count, tacudaRequest requests[]);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif 
