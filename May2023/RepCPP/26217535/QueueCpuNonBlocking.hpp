

#pragma once

#include <alpaka/event/EventCpu.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>

namespace alpaka
{
using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
}
