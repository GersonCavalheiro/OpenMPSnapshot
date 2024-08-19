

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/meta/DependentFalseType.hpp>
#    include <alpaka/queue/Traits.hpp>
#    include <alpaka/wait/Traits.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>
#    else
#        include <alpaka/core/Hip.hpp>
#    endif

#    include <alpaka/core/CallbackThread.hpp>

#    include <condition_variable>
#    include <functional>
#    include <future>
#    include <memory>
#    include <mutex>
#    include <thread>

namespace alpaka
{
template<typename TApi>
class EventUniformCudaHipRt;

namespace uniform_cuda_hip::detail
{
template<typename TApi>
class QueueUniformCudaHipRtImpl final
{
public:
ALPAKA_FN_HOST QueueUniformCudaHipRtImpl(DevUniformCudaHipRt<TApi> const& dev)
: m_dev(dev)
, m_UniformCudaHipQueue()
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::setDevice(m_dev.getNativeHandle()));


ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
TApi::streamCreateWithFlags(&m_UniformCudaHipQueue, TApi::streamNonBlocking));
}
QueueUniformCudaHipRtImpl(QueueUniformCudaHipRtImpl&&) = default;
auto operator=(QueueUniformCudaHipRtImpl&&) -> QueueUniformCudaHipRtImpl& = delete;
ALPAKA_FN_HOST ~QueueUniformCudaHipRtImpl()
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(TApi::streamDestroy(m_UniformCudaHipQueue));
}

[[nodiscard]] auto getNativeHandle() const noexcept
{
return m_UniformCudaHipQueue;
}

public:
DevUniformCudaHipRt<TApi> const m_dev; 
core::CallbackThread m_callbackThread;

private:
typename TApi::Stream_t m_UniformCudaHipQueue;
};

template<typename TApi, bool TBlocking>
class QueueUniformCudaHipRt
: public concepts::Implements<ConceptCurrentThreadWaitFor, QueueUniformCudaHipRt<TApi, TBlocking>>
, public concepts::Implements<ConceptQueue, QueueUniformCudaHipRt<TApi, TBlocking>>
, public concepts::Implements<ConceptGetDev, QueueUniformCudaHipRt<TApi, TBlocking>>
{
public:
ALPAKA_FN_HOST QueueUniformCudaHipRt(DevUniformCudaHipRt<TApi> const& dev)
: m_spQueueImpl(std::make_shared<QueueUniformCudaHipRtImpl<TApi>>(dev))
{
}
ALPAKA_FN_HOST auto operator==(QueueUniformCudaHipRt const& rhs) const -> bool
{
return (m_spQueueImpl == rhs.m_spQueueImpl);
}
ALPAKA_FN_HOST auto operator!=(QueueUniformCudaHipRt const& rhs) const -> bool
{
return !((*this) == rhs);
}

[[nodiscard]] auto getNativeHandle() const noexcept
{
return m_spQueueImpl->getNativeHandle();
}
auto getCallbackThread() -> core::CallbackThread&
{
return m_spQueueImpl->m_callbackThread;
}

public:
std::shared_ptr<QueueUniformCudaHipRtImpl<TApi>> m_spQueueImpl;
};
} 

namespace trait
{
template<typename TApi, bool TBlocking>
struct GetDev<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
{
ALPAKA_FN_HOST static auto getDev(
uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue)
-> DevUniformCudaHipRt<TApi>
{
return queue.m_spQueueImpl->m_dev;
}
};

template<typename TApi, bool TBlocking>
struct Empty<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
{
ALPAKA_FN_HOST static auto empty(
uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue) -> bool
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

typename TApi::Error_t ret = TApi::success;
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_IGNORE(
ret = TApi::streamQuery(queue.getNativeHandle()),
TApi::errorNotReady);
return (ret == TApi::success);
}
};

template<typename TApi, bool TBlocking>
struct CurrentThreadWaitFor<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(
uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::streamSynchronize(queue.getNativeHandle()));
}
};

template<typename TApi, bool TBlocking>
struct DevType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
{
using type = DevUniformCudaHipRt<TApi>;
};

template<typename TApi, bool TBlocking>
struct EventType<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
{
using type = EventUniformCudaHipRt<TApi>;
};

template<typename TApi, bool TBlocking, typename TTask>
struct Enqueue<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>, TTask>
{
enum class CallbackState
{
enqueued,
notified,
finished,
};

struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
{
std::mutex m_mutex;
std::condition_variable m_event;
CallbackState m_state = CallbackState::enqueued;
};

ALPAKA_FN_HOST static void uniformCudaHipRtHostFunc(void* arg)
{
auto const spCallbackSynchronizationData
= reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

{
std::unique_lock<std::mutex> lock(spCallbackSynchronizationData->m_mutex);
spCallbackSynchronizationData->m_state = CallbackState::notified;
}
spCallbackSynchronizationData->m_event.notify_one();

std::unique_lock<std::mutex> lock(spCallbackSynchronizationData->m_mutex);
if(spCallbackSynchronizationData->m_state != CallbackState::finished)
{
spCallbackSynchronizationData->m_event.wait(
lock,
[&spCallbackSynchronizationData]()
{ return spCallbackSynchronizationData->m_state == CallbackState::finished; });
}
}

ALPAKA_FN_HOST static auto enqueue(
uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>& queue,
TTask const& task) -> void
{
auto spCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::launchHostFunc(
queue.getNativeHandle(),
uniformCudaHipRtHostFunc,
spCallbackSynchronizationData.get()));

auto f = queue.getCallbackThread().submit(std::packaged_task<void()>(
[spCallbackSynchronizationData, task]()
{
{
std::unique_lock<std::mutex> lock(spCallbackSynchronizationData->m_mutex);
if(spCallbackSynchronizationData->m_state != CallbackState::notified)
{
spCallbackSynchronizationData->m_event.wait(
lock,
[&spCallbackSynchronizationData]()
{ return spCallbackSynchronizationData->m_state == CallbackState::notified; });
}

task();

spCallbackSynchronizationData->m_state = CallbackState::finished;
}
spCallbackSynchronizationData->m_event.notify_one();
}));

if constexpr(TBlocking)
{
f.wait();
}
}
};

template<typename TApi, bool TBlocking>
struct NativeHandle<uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking>>
{
[[nodiscard]] static auto getNativeHandle(
uniform_cuda_hip::detail::QueueUniformCudaHipRt<TApi, TBlocking> const& queue)
{
return queue.getNativeHandle();
}
};
} 
} 

#endif
