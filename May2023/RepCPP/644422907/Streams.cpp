

#include <TAHIP.h>

#include "common/StreamPool.hpp"
#include "common/Environment.hpp"
#include "common/TaskingModel.hpp"
#include "common/util/ErrorHandler.hpp"

using namespace tahip;

#pragma GCC visibility push(default)

extern "C" {

hipError_t
tahipCreateStreams(size_t count)
{
if (count == TAHIP_STREAMS_AUTO)
count = TaskingModel::getNumCPUs();
assert(count > 0);

StreamPool::initialize(count);

return hipSuccess;
}

hipError_t
tahipDestroyStreams()
{
StreamPool::finalize();

return hipSuccess;
}

hipError_t
tahipGetStream(hipStream_t *stream)
{
assert(stream != nullptr);

*stream = StreamPool::getStream(TaskingModel::getCurrentCPU());

return hipSuccess;
}

hipError_t
tahipReturnStream(hipStream_t)
{
return hipSuccess;
}

hipError_t
tahipSynchronizeStreamAsync(hipStream_t stream)
{
RequestManager::generateRequest(stream, true);

return hipSuccess;
}

} 

#pragma GCC visibility pop
