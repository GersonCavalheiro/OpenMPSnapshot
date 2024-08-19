

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/event/EventCpu.hpp>
#include <alpaka/wait/Traits.hpp>

namespace alpaka
{
namespace trait
{
template<>
struct CurrentThreadWaitFor<DevCpu>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(DevCpu const& dev) -> void
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

generic::currentThreadWaitForDevice(dev);
}
};
} 
} 
