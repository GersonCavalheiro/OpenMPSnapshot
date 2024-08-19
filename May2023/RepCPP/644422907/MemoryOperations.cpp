

#include <TAHIP.h>

#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tahip;

#pragma GCC visibility push(default)

extern "C" {

hipError_t
tahipMemcpyAsync(
void *dst, const void *src, size_t sizeBytes,
enum hipMemcpyKind kind, hipStream_t stream,
tahipRequest *requestPtr)
{
hipError_t eret;
eret = hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
if (eret != hipSuccess)
return eret;

Request *request = RequestManager::generateRequest(stream, (requestPtr == nullptr));
assert(request != nullptr);

if (requestPtr != nullptr)
*requestPtr = request;

return hipSuccess;
}

hipError_t
tahipMemsetAsync(
void *devPtr, int value, size_t sizeBytes, hipStream_t stream,
tahipRequest *requestPtr)
{
hipError_t eret;
eret = hipMemsetAsync(devPtr, value, sizeBytes, stream);
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
