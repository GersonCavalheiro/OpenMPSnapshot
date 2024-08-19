

#include <TAHIP.h>

#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tahip;

#pragma GCC visibility push(default)

extern "C" {

hipblasStatus_t
tahipblasGemmEx(hipblasHandle_t handle,
hipblasOperation_t transa, hipblasOperation_t transb, int m,
int n, int k, const void *alpha, const void *matA, hipblasDatatype_t Atype,
int lda, const void *matB, hipblasDatatype_t Btype, int ldb,
const void *beta, void *matC, hipblasDatatype_t Ctype, int ldc, hipblasDatatype_t computeType,
hipblasGemmAlgo_t algo, hipStream_t stream, tahipRequest *requestPtr)
{
hipblasStatus_t stat;
stat = hipblasSetStream(handle, stream);
if (stat != HIPBLAS_STATUS_SUCCESS)
return stat;

stat = hipblasGemmEx(handle, transa, transb,
m, n, k, alpha, matA, Atype, lda, matB, Btype, ldb,
beta, matC, Ctype, ldc, computeType, algo);
if (stat != HIPBLAS_STATUS_SUCCESS)
return stat;

Request *request = RequestManager::generateRequest(stream, (requestPtr == nullptr));
assert(request != nullptr);

if (requestPtr != nullptr)
*requestPtr = request;

return HIPBLAS_STATUS_SUCCESS;
}

} 

#pragma GCC visibility pop
