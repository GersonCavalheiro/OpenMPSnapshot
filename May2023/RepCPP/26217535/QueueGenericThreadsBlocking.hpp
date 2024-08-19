

#pragma once

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/wait/Traits.hpp>

#include <atomic>
#include <memory>
#include <mutex>

namespace alpaka
{
template<typename TDev>
class EventGenericThreads;

namespace generic
{
namespace detail
{
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
template<typename TDev>
class QueueGenericThreadsBlockingImpl final : public IGenericThreadsQueue<TDev>
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
{
public:
explicit QueueGenericThreadsBlockingImpl(TDev dev) noexcept
: m_dev(std::move(dev))
, m_bCurrentlyExecutingTask(false)
{
}
QueueGenericThreadsBlockingImpl(QueueGenericThreadsBlockingImpl<TDev> const&) = delete;
auto operator=(QueueGenericThreadsBlockingImpl<TDev> const&)
-> QueueGenericThreadsBlockingImpl<TDev>& = delete;

void enqueue(EventGenericThreads<TDev>& ev) final
{
alpaka::enqueue(*this, ev);
}

void wait(EventGenericThreads<TDev> const& ev) final
{
alpaka::wait(*this, ev);
}

public:
TDev const m_dev; 
std::mutex mutable m_mutex;
std::atomic<bool> m_bCurrentlyExecutingTask;
};
} 
} 

template<typename TDev>
class QueueGenericThreadsBlocking final
: public concepts::Implements<ConceptCurrentThreadWaitFor, QueueGenericThreadsBlocking<TDev>>
, public concepts::Implements<ConceptQueue, QueueGenericThreadsBlocking<TDev>>
, public concepts::Implements<ConceptGetDev, QueueGenericThreadsBlocking<TDev>>
{
public:
explicit QueueGenericThreadsBlocking(TDev const& dev)
: m_spQueueImpl(std::make_shared<generic::detail::QueueGenericThreadsBlockingImpl<TDev>>(dev))
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

dev.registerQueue(m_spQueueImpl);
}
auto operator==(QueueGenericThreadsBlocking<TDev> const& rhs) const -> bool
{
return (m_spQueueImpl == rhs.m_spQueueImpl);
}
auto operator!=(QueueGenericThreadsBlocking<TDev> const& rhs) const -> bool
{
return !((*this) == rhs);
}

public:
std::shared_ptr<generic::detail::QueueGenericThreadsBlockingImpl<TDev>> m_spQueueImpl;
};

namespace trait
{
template<typename TDev>
struct DevType<QueueGenericThreadsBlocking<TDev>>
{
using type = TDev;
};
template<typename TDev>
struct GetDev<QueueGenericThreadsBlocking<TDev>>
{
ALPAKA_FN_HOST static auto getDev(QueueGenericThreadsBlocking<TDev> const& queue) -> TDev
{
return queue.m_spQueueImpl->m_dev;
}
};

template<typename TDev>
struct EventType<QueueGenericThreadsBlocking<TDev>>
{
using type = EventGenericThreads<TDev>;
};

template<typename TDev, typename TTask>
struct Enqueue<QueueGenericThreadsBlocking<TDev>, TTask>
{
ALPAKA_FN_HOST static auto enqueue(QueueGenericThreadsBlocking<TDev>& queue, TTask const& task) -> void
{
std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);

queue.m_spQueueImpl->m_bCurrentlyExecutingTask = true;

task();

queue.m_spQueueImpl->m_bCurrentlyExecutingTask = false;
}
};
template<typename TDev>
struct Empty<QueueGenericThreadsBlocking<TDev>>
{
ALPAKA_FN_HOST static auto empty(QueueGenericThreadsBlocking<TDev> const& queue) -> bool
{
return !queue.m_spQueueImpl->m_bCurrentlyExecutingTask;
}
};

template<typename TDev>
struct CurrentThreadWaitFor<QueueGenericThreadsBlocking<TDev>>
{
ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueGenericThreadsBlocking<TDev> const& queue) -> void
{
std::lock_guard<std::mutex> lk(queue.m_spQueueImpl->m_mutex);
}
};
} 
} 

#include <alpaka/event/EventGenericThreads.hpp>
