

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/mem/buf/Traits.hpp>
#    include <alpaka/pltf/Traits.hpp>
#    include <alpaka/queue/Properties.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/traits/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <cstddef>
#    include <string>
#    include <vector>

namespace alpaka
{
namespace trait
{
template<typename TPltf, typename TSfinae>
struct GetDevByIdx;
}

namespace uniform_cuda_hip::detail
{
template<typename TApi, bool TBlocking>
class QueueUniformCudaHipRt;
}

template<typename TApi>
using QueueUniformCudaHipRtBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, true>;

template<typename TApi>
using QueueUniformCudaHipRtNonBlocking = uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, false>;

template<typename TApi>
class PltfUniformCudaHipRt;

template<typename TApi, typename TElem, typename TDim, typename TIdx>
class BufUniformCudaHipRt;

template<typename TApi>
class DevUniformCudaHipRt
: public concepts::Implements<ConceptCurrentThreadWaitFor, DevUniformCudaHipRt<TApi>>
, public concepts::Implements<ConceptDev, DevUniformCudaHipRt<TApi>>
{
friend struct trait::GetDevByIdx<PltfUniformCudaHipRt<TApi>>;

protected:
DevUniformCudaHipRt() = default;

public:
ALPAKA_FN_HOST auto operator==(DevUniformCudaHipRt const& rhs) const -> bool
{
return m_iDevice == rhs.m_iDevice;
}
ALPAKA_FN_HOST auto operator!=(DevUniformCudaHipRt const& rhs) const -> bool
{
return !((*this) == rhs);
}

[[nodiscard]] auto getNativeHandle() const noexcept -> int
{
return m_iDevice;
}

private:
DevUniformCudaHipRt(int iDevice) : m_iDevice(iDevice)
{
}
int m_iDevice;
};

namespace trait
{
template<typename TApi>
struct GetName<DevUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto getName(DevUniformCudaHipRt<TApi> const& dev) -> std::string
{
typename TApi::DeviceProp_t devProp;
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&devProp, dev.getNativeHandle()));

return std::string(devProp.name);
}
};

template<typename TApi>
struct GetMemBytes<DevUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto getMemBytes(DevUniformCudaHipRt<TApi> const& dev) -> std::size_t
{
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

std::size_t freeInternal(0u);
std::size_t totalInternal(0u);

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memGetInfo(&freeInternal, &totalInternal));

return totalInternal;
}
};

template<typename TApi>
struct GetFreeMemBytes<DevUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto getFreeMemBytes(DevUniformCudaHipRt<TApi> const& dev) -> std::size_t
{
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

std::size_t freeInternal(0u);
std::size_t totalInternal(0u);

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::memGetInfo(&freeInternal, &totalInternal));

return freeInternal;
}
};

template<typename TApi>
struct GetWarpSizes<DevUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto getWarpSizes(DevUniformCudaHipRt<TApi> const& dev) -> std::vector<std::size_t>
{
typename TApi::DeviceProp_t devProp;
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getDeviceProperties(&devProp, dev.getNativeHandle()));

return {static_cast<std::size_t>(devProp.warpSize)};
}
};

template<typename TApi>
struct Reset<DevUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto reset(DevUniformCudaHipRt<TApi> const& dev) -> void
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceReset());
}
};

template<typename TApi>
struct NativeHandle<DevUniformCudaHipRt<TApi>>
{
[[nodiscard]] static auto getNativeHandle(DevUniformCudaHipRt<TApi> const& dev)
{
return dev.getNativeHandle();
}
};

template<typename TApi, typename TElem, typename TDim, typename TIdx>
struct BufType<DevUniformCudaHipRt<TApi>, TElem, TDim, TIdx>
{
using type = BufUniformCudaHipRt<TApi, TElem, TDim, TIdx>;
};

template<typename TApi>
struct PltfType<DevUniformCudaHipRt<TApi>>
{
using type = PltfUniformCudaHipRt<TApi>;
};

template<typename TApi>
struct CurrentThreadWaitFor<DevUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(DevUniformCudaHipRt<TApi> const& dev) -> void
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::deviceSynchronize());
}
};

template<typename TApi>
struct QueueType<DevUniformCudaHipRt<TApi>, Blocking>
{
using type = QueueUniformCudaHipRtBlocking<TApi>;
};

template<typename TApi>
struct QueueType<DevUniformCudaHipRt<TApi>, NonBlocking>
{
using type = QueueUniformCudaHipRtNonBlocking<TApi>;
};
} 
} 

#endif
