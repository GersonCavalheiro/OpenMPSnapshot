

#pragma once

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Utility.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#include <alpaka/wait/Traits.hpp>

#include <condition_variable>
#include <future>
#include <mutex>
#include <utility>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

namespace alpaka
{
namespace generic::detail
{
template<typename TDev>
class EventGenericThreadsImpl final
: public concepts::Implements<ConceptCurrentThreadWaitFor, EventGenericThreadsImpl<TDev>>
{
public:
EventGenericThreadsImpl(TDev dev) noexcept : m_dev(std::move(dev))
{
}
EventGenericThreadsImpl(EventGenericThreadsImpl<TDev> const&) = delete;
auto operator=(EventGenericThreadsImpl<TDev> const&) -> EventGenericThreadsImpl<TDev>& = delete;

auto isReady() noexcept -> bool
{
return (m_LastReadyEnqueueCount == m_enqueueCount);
}

auto wait(std::size_t const& enqueueCount, std::unique_lock<std::mutex>& lk) const noexcept -> void
{
ALPAKA_ASSERT(enqueueCount <= m_enqueueCount);

while(enqueueCount > m_LastReadyEnqueueCount)
{
auto future = m_future;
lk.unlock();
future.get();
lk.lock();
}
}

TDev const m_dev; 

std::mutex mutable m_mutex; 
std::shared_future<void> m_future; 
std::size_t m_enqueueCount = 0u; 
std::size_t m_LastReadyEnqueueCount = 0u; 
};
} 

template<typename TDev>
class EventGenericThreads final
: public concepts::Implements<ConceptCurrentThreadWaitFor, EventGenericThreads<TDev>>
, public concepts::Implements<ConceptGetDev, EventGenericThreads<TDev>>
{
public:
EventGenericThreads(TDev const& dev, [[maybe_unused]] bool bBusyWaiting = true)
: m_spEventImpl(std::make_shared<generic::detail::EventGenericThreadsImpl<TDev>>(dev))
{
}
auto operator==(EventGenericThreads<TDev> const& rhs) const -> bool
{
return (m_spEventImpl == rhs.m_spEventImpl);
}
auto operator!=(EventGenericThreads<TDev> const& rhs) const -> bool
{
return !((*this) == rhs);
}

public:
std::shared_ptr<generic::detail::EventGenericThreadsImpl<TDev>> m_spEventImpl;
};
namespace trait
{
template<typename TDev>
struct DevType<EventGenericThreads<TDev>>
{
using type = TDev;
};
template<typename TDev>
struct GetDev<EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto getDev(EventGenericThreads<TDev> const& event) -> TDev
{
return event.m_spEventImpl->m_dev;
}
};

template<typename TDev>
struct IsComplete<EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto isComplete(EventGenericThreads<TDev> const& event) -> bool
{
std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

return event.m_spEventImpl->isReady();
}
};

template<typename TDev>
struct Enqueue<alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto enqueue(
[[maybe_unused]] alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>& queueImpl,
EventGenericThreads<TDev>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto spEventImpl = event.m_spEventImpl;

std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

++spEventImpl->m_enqueueCount;

auto const enqueueCount = spEventImpl->m_enqueueCount;

spEventImpl->m_future = queueImpl.m_workerThread->enqueueTask(
[spEventImpl, enqueueCount]()
{
std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);

if(enqueueCount == spEventImpl->m_enqueueCount)
{
spEventImpl->m_LastReadyEnqueueCount = spEventImpl->m_enqueueCount;
}
});
}
};
template<typename TDev>
struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueGenericThreadsNonBlocking<TDev>& queue,
EventGenericThreads<TDev>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

alpaka::enqueue(*queue.m_spQueueImpl, event);
}
};
template<typename TDev>
struct Enqueue<alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto enqueue(
alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>& queueImpl,
EventGenericThreads<TDev>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

std::promise<void> promise;
{
std::lock_guard<std::mutex> lk(queueImpl.m_mutex);

queueImpl.m_bCurrentlyExecutingTask = true;

auto& eventImpl(*event.m_spEventImpl);

{
std::lock_guard<std::mutex> evLk(eventImpl.m_mutex);

++eventImpl.m_enqueueCount;
eventImpl.m_LastReadyEnqueueCount = eventImpl.m_enqueueCount;

eventImpl.m_future = promise.get_future();
}

queueImpl.m_bCurrentlyExecutingTask = false;
}
promise.set_value();
}
};
template<typename TDev>
struct Enqueue<QueueGenericThreadsBlocking<TDev>, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueGenericThreadsBlocking<TDev>& queue,
EventGenericThreads<TDev>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

alpaka::enqueue(*queue.m_spQueueImpl, event);
}
};
} 
namespace trait
{
namespace generic
{
template<typename TDev>
ALPAKA_FN_HOST auto currentThreadWaitForDevice(TDev const& dev) -> void
{
auto vQueues = dev.getAllQueues();
std::vector<EventGenericThreads<TDev>> vEvents;
for(auto&& spQueue : vQueues)
{
vEvents.emplace_back(dev);
spQueue->enqueue(vEvents.back());
}

for(auto&& event : vEvents)
{
wait(event);
}
}
} 

template<typename TDev>
struct CurrentThreadWaitFor<EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(EventGenericThreads<TDev> const& event) -> void
{
wait(*event.m_spEventImpl);
}
};
template<typename TDev>
struct CurrentThreadWaitFor<alpaka::generic::detail::EventGenericThreadsImpl<TDev>>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(
alpaka::generic::detail::EventGenericThreadsImpl<TDev> const& eventImpl) -> void
{
std::unique_lock<std::mutex> lk(eventImpl.m_mutex);

auto const enqueueCount = eventImpl.m_enqueueCount;
eventImpl.wait(enqueueCount, lk);
}
};
template<typename TDev>
struct WaiterWaitFor<
alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>,
EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(
alpaka::generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>& queueImpl,
EventGenericThreads<TDev> const& event) -> void
{
auto spEventImpl = event.m_spEventImpl;

std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

if(!spEventImpl->isReady())
{
auto const enqueueCount = spEventImpl->m_enqueueCount;

queueImpl.m_workerThread->enqueueTask(
[spEventImpl, enqueueCount]()
{
std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);
spEventImpl->wait(enqueueCount, lk2);
});
}
}
};
template<typename TDev>
struct WaiterWaitFor<QueueGenericThreadsNonBlocking<TDev>, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(
QueueGenericThreadsNonBlocking<TDev>& queue,
EventGenericThreads<TDev> const& event) -> void
{
wait(*queue.m_spQueueImpl, event);
}
};
template<typename TDev>
struct WaiterWaitFor<alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(
alpaka::generic::detail::QueueGenericThreadsBlockingImpl<TDev>& ,
EventGenericThreads<TDev> const& event) -> void
{
wait(*event.m_spEventImpl);
}
};
template<typename TDev>
struct WaiterWaitFor<QueueGenericThreadsBlocking<TDev>, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(
QueueGenericThreadsBlocking<TDev>& queue,
EventGenericThreads<TDev> const& event) -> void
{
wait(*queue.m_spQueueImpl, event);
}
};
template<typename TDev>
struct WaiterWaitFor<TDev, EventGenericThreads<TDev>>
{
ALPAKA_FN_HOST static auto waiterWaitFor(TDev& dev, EventGenericThreads<TDev> const& event) -> void
{
auto vspQueues(dev.getAllQueues());

for(auto&& spQueue : vspQueues)
{
spQueue->wait(event);
}
}
};

template<typename TDev>
struct CurrentThreadWaitFor<QueueGenericThreadsNonBlocking<TDev>>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueGenericThreadsNonBlocking<TDev> const& queue) -> void
{
EventGenericThreads<TDev> event(getDev(queue));
alpaka::enqueue(const_cast<QueueGenericThreadsNonBlocking<TDev>&>(queue), event);
wait(event);
}
};
} 
} 
