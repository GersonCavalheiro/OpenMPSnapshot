

#pragma once

#include <alpaka/event/EventCpu.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>

namespace alpaka
{
using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;
}
