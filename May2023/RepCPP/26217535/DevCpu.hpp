

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/common/QueueRegistry.hpp>
#include <alpaka/dev/cpu/SysInfo.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/queue/Properties.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/traits/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace alpaka
{
class DevCpu;
namespace cpu
{
using ICpuQueue = IGenericThreadsQueue<DevCpu>;
}
namespace trait
{
template<typename TPltf, typename TSfinae>
struct GetDevByIdx;
}
class PltfCpu;

namespace cpu::detail
{
using DevCpuImpl = alpaka::detail::QueueRegistry<cpu::ICpuQueue>;
} 

class DevCpu
: public concepts::Implements<ConceptCurrentThreadWaitFor, DevCpu>
, public concepts::Implements<ConceptDev, DevCpu>
{
friend struct trait::GetDevByIdx<PltfCpu>;

protected:
DevCpu() : m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
{
}

public:
auto operator==(DevCpu const&) const -> bool
{
return true;
}
auto operator!=(DevCpu const& rhs) const -> bool
{
return !((*this) == rhs);
}

[[nodiscard]] ALPAKA_FN_HOST auto getAllQueues() const -> std::vector<std::shared_ptr<cpu::ICpuQueue>>
{
return m_spDevCpuImpl->getAllExistingQueues();
}

ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<cpu::ICpuQueue> spQueue) const -> void
{
m_spDevCpuImpl->registerQueue(spQueue);
}

auto registerCleanup(cpu::detail::DevCpuImpl::CleanerFunctor c) const -> void
{
m_spDevCpuImpl->registerCleanup(c);
}

[[nodiscard]] auto getNativeHandle() const noexcept
{
return 0;
}

private:
std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
};

namespace trait
{
template<>
struct GetName<DevCpu>
{
ALPAKA_FN_HOST static auto getName(DevCpu const& ) -> std::string
{
return cpu::detail::getCpuName();
}
};

template<>
struct GetMemBytes<DevCpu>
{
ALPAKA_FN_HOST static auto getMemBytes(DevCpu const& ) -> std::size_t
{
return cpu::detail::getTotalGlobalMemSizeBytes();
}
};

template<>
struct GetFreeMemBytes<DevCpu>
{
ALPAKA_FN_HOST static auto getFreeMemBytes(DevCpu const& ) -> std::size_t
{
return cpu::detail::getFreeGlobalMemSizeBytes();
}
};

template<>
struct GetWarpSizes<DevCpu>
{
ALPAKA_FN_HOST static auto getWarpSizes(DevCpu const& ) -> std::vector<std::size_t>
{
return {1u};
}
};

template<>
struct Reset<DevCpu>
{
ALPAKA_FN_HOST static auto reset(DevCpu const& ) -> void
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;
}
};

template<>
struct NativeHandle<DevCpu>
{
[[nodiscard]] static auto getNativeHandle(DevCpu const& dev)
{
return dev.getNativeHandle();
}
};
} 

template<typename TElem, typename TDim, typename TIdx>
class BufCpu;

namespace trait
{
template<typename TElem, typename TDim, typename TIdx>
struct BufType<DevCpu, TElem, TDim, TIdx>
{
using type = BufCpu<TElem, TDim, TIdx>;
};

template<>
struct PltfType<DevCpu>
{
using type = PltfCpu;
};
} 
using QueueCpuNonBlocking = QueueGenericThreadsNonBlocking<DevCpu>;
using QueueCpuBlocking = QueueGenericThreadsBlocking<DevCpu>;

namespace trait
{
template<>
struct QueueType<DevCpu, Blocking>
{
using type = QueueCpuBlocking;
};

template<>
struct QueueType<DevCpu, NonBlocking>
{
using type = QueueCpuNonBlocking;
};
} 
} 
