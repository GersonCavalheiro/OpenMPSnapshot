

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <alpaka/queue/QueueUniformCudaHipRtBlocking.hpp>
#    include <alpaka/queue/QueueUniformCudaHipRtNonBlocking.hpp>

#    include <functional>
#    include <memory>
#    include <stdexcept>

namespace alpaka
{
namespace uniform_cuda_hip::detail
{
template<typename TApi>
class EventUniformCudaHipImpl final
{
public:
ALPAKA_FN_HOST EventUniformCudaHipImpl(DevUniformCudaHipRt<TApi> const& dev, bool bBusyWait)
: m_dev(dev)
, m_UniformCudaHipEvent()
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_dev.getNativeHandle()));

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventCreateWithFlags(
&m_UniformCudaHipEvent,
(bBusyWait ? TApi::eventDefault : TApi::eventBlockingSync) | TApi::eventDisableTiming));
}
EventUniformCudaHipImpl(EventUniformCudaHipImpl const&) = delete;
auto operator=(EventUniformCudaHipImpl const&) -> EventUniformCudaHipImpl& = delete;
ALPAKA_FN_HOST ~EventUniformCudaHipImpl()
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::eventDestroy(m_UniformCudaHipEvent));
}

[[nodiscard]] auto getNativeHandle() const noexcept
{
return m_UniformCudaHipEvent;
}

public:
DevUniformCudaHipRt<TApi> const m_dev; 

private:
typename TApi::Event_t m_UniformCudaHipEvent;
};
} 

template<typename TApi>
class EventUniformCudaHipRt final
: public concepts::Implements<ConceptCurrentThreadWaitFor, EventUniformCudaHipRt<TApi>>
, public concepts::Implements<ConceptGetDev, EventUniformCudaHipRt<TApi>>
{
public:
ALPAKA_FN_HOST EventUniformCudaHipRt<TApi>(DevUniformCudaHipRt<TApi> const& dev, bool bBusyWait = true)
: m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventUniformCudaHipImpl<TApi>>(dev, bBusyWait))
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
}
ALPAKA_FN_HOST auto operator==(EventUniformCudaHipRt<TApi> const& rhs) const -> bool
{
return (m_spEventImpl == rhs.m_spEventImpl);
}
ALPAKA_FN_HOST auto operator!=(EventUniformCudaHipRt<TApi> const& rhs) const -> bool
{
return !((*this) == rhs);
}

[[nodiscard]] auto getNativeHandle() const noexcept
{
return m_spEventImpl->getNativeHandle();
}

public:
std::shared_ptr<uniform_cuda_hip::detail::EventUniformCudaHipImpl<TApi>> m_spEventImpl;
};
namespace trait
{
template<typename TApi>
struct DevType<EventUniformCudaHipRt<TApi>>
{
using type = DevUniformCudaHipRt<TApi>;
};
template<typename TApi>
struct GetDev<EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto getDev(EventUniformCudaHipRt<TApi> const& event) -> DevUniformCudaHipRt<TApi>
{
return event.m_spEventImpl->m_dev;
}
};

template<typename TApi>
struct IsComplete<EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto isComplete(EventUniformCudaHipRt<TApi> const& event) -> bool
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

typename TApi::Error_t ret = TApi::success;
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
ret = TApi::eventQuery(event.getNativeHandle()),
TApi::errorNotReady);
return (ret == TApi::success);
}
};

template<typename TApi>
struct Enqueue<QueueUniformCudaHipRtNonBlocking<TApi>, EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueUniformCudaHipRtNonBlocking<TApi>& queue,
EventUniformCudaHipRt<TApi>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventRecord(event.getNativeHandle(), queue.getNativeHandle()));
}
};
template<typename TApi>
struct Enqueue<QueueUniformCudaHipRtBlocking<TApi>, EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueUniformCudaHipRtBlocking<TApi>& queue,
EventUniformCudaHipRt<TApi>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventRecord(event.getNativeHandle(), queue.getNativeHandle()));
}
};

template<typename TApi>
struct CurrentThreadWaitFor<EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(EventUniformCudaHipRt<TApi> const& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::eventSynchronize(event.getNativeHandle()));
}
};
template<typename TApi>
struct WaiterWaitFor<QueueUniformCudaHipRtNonBlocking<TApi>, EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(
QueueUniformCudaHipRtNonBlocking<TApi>& queue,
EventUniformCudaHipRt<TApi> const& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
TApi::streamWaitEvent(queue.getNativeHandle(), event.getNativeHandle(), 0));
}
};
template<typename TApi>
struct WaiterWaitFor<QueueUniformCudaHipRtBlocking<TApi>, EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(
QueueUniformCudaHipRtBlocking<TApi>& queue,
EventUniformCudaHipRt<TApi> const& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
TApi::streamWaitEvent(queue.getNativeHandle(), event.getNativeHandle(), 0));
}
};
template<typename TApi>
struct WaiterWaitFor<DevUniformCudaHipRt<TApi>, EventUniformCudaHipRt<TApi>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(
DevUniformCudaHipRt<TApi>& dev,
EventUniformCudaHipRt<TApi> const& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(dev.getNativeHandle()));

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamWaitEvent(nullptr, event.getNativeHandle(), 0));
}
};
template<typename TApi>
struct NativeHandle<EventUniformCudaHipRt<TApi>>
{
[[nodiscard]] static auto getNativeHandle(EventUniformCudaHipRt<TApi> const& event)
{
return event.getNativeHandle();
}
};
} 
} 

#endif
