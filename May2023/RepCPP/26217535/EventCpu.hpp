

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/event/EventGenericThreads.hpp>

namespace alpaka
{
using EventCpu = EventGenericThreads<DevCpu>;
}
