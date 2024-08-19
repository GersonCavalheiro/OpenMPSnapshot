

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>

namespace alpaka::cpu
{
using ICpuQueue = IGenericThreadsQueue<DevCpu>;
} 
