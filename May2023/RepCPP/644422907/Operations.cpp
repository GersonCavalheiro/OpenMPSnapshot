

#include <TAHIP.h>

#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tahip;

#pragma GCC visibility push(default)

extern "C" {

hipError_t
tahipLaunchKernel(
const void* func, dim3 gridDim, dim3 blockDim, 
void** args, size_t sharedMem, hipStream_t stream,
tahipRequest *requestPtr)
{

hipError_t eret;

eret = hipLaunchKernel(func, gridDim, blockDim, args,
sharedMem, stream);
if (eret != hipSuccess)
return eret;

Request *request = RequestManager::generateRequest(stream, (requestPtr == nullptr));
assert(request != nullptr);

if (requestPtr != nullptr)
*requestPtr = request;

return hipSuccess;
}

} 

#pragma GCC visibility pop
