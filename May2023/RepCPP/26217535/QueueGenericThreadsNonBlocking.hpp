

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/ConcurrentExecPool.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/wait/Traits.hpp>

#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <type_traits>

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
class QueueGenericThreadsNonBlockingImpl final : public IGenericThreadsQueue<TDev>
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
{
private:
using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
std::size_t,
std::thread, 
std::promise, 
void, 
std::mutex, 
std::condition_variable, 
false>; 

public:
explicit QueueGenericThreadsNonBlockingImpl(TDev dev)
: m_dev(std::move(dev))
, m_workerThread(std::make_shared<ThreadPool>(1u))
{
}
QueueGenericThreadsNonBlockingImpl(QueueGenericThreadsNonBlockingImpl<TDev> const&) = delete;
QueueGenericThreadsNonBlockingImpl(QueueGenericThreadsNonBlockingImpl<TDev>&&) = delete;
auto operator=(QueueGenericThreadsNonBlockingImpl<TDev> const&)
-> QueueGenericThreadsNonBlockingImpl<TDev>& = delete;
auto operator=(QueueGenericThreadsNonBlockingImpl&&)
-> QueueGenericThreadsNonBlockingImpl<TDev>& = delete;
~QueueGenericThreadsNonBlockingImpl() override
{
m_dev.registerCleanup(
[pool = std::weak_ptr<ThreadPool>(m_workerThread)]() noexcept
{
if(auto s = pool.lock())
std::ignore = s->takeDetachHandle(); 
});
auto* wt = m_workerThread.get();
wt->detach(std::move(m_workerThread));
}

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

std::shared_ptr<ThreadPool> m_workerThread;
};
} 
} 

template<typename TDev>
class QueueGenericThreadsNonBlocking final
: public concepts::Implements<ConceptCurrentThreadWaitFor, QueueGenericThreadsNonBlocking<TDev>>
, public concepts::Implements<ConceptQueue, QueueGenericThreadsNonBlocking<TDev>>
, public concepts::Implements<ConceptGetDev, QueueGenericThreadsNonBlocking<TDev>>
{
public:
explicit QueueGenericThreadsNonBlocking(TDev const& dev)
: m_spQueueImpl(std::make_shared<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>>(dev))
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

dev.registerQueue(m_spQueueImpl);
}
auto operator==(QueueGenericThreadsNonBlocking<TDev> const& rhs) const -> bool
{
return (m_spQueueImpl == rhs.m_spQueueImpl);
}
auto operator!=(QueueGenericThreadsNonBlocking<TDev> const& rhs) const -> bool
{
return !((*this) == rhs);
}

public:
std::shared_ptr<generic::detail::QueueGenericThreadsNonBlockingImpl<TDev>> m_spQueueImpl;
};

namespace trait
{
template<typename TDev>
struct DevType<QueueGenericThreadsNonBlocking<TDev>>
{
using type = TDev;
};
template<typename TDev>
struct GetDev<QueueGenericThreadsNonBlocking<TDev>>
{
ALPAKA_FN_HOST static auto getDev(QueueGenericThreadsNonBlocking<TDev> const& queue) -> TDev
{
return queue.m_spQueueImpl->m_dev;
}
};

template<typename TDev>
struct EventType<QueueGenericThreadsNonBlocking<TDev>>
{
using type = EventGenericThreads<TDev>;
};

template<typename TDev, typename TTask>
struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, TTask>
{
ALPAKA_FN_HOST static auto enqueue(QueueGenericThreadsNonBlocking<TDev>& queue, TTask const& task) -> void
{
queue.m_spQueueImpl->m_workerThread->enqueueTask(task);
}
};
template<typename TDev>
struct Empty<QueueGenericThreadsNonBlocking<TDev>>
{
ALPAKA_FN_HOST static auto empty(QueueGenericThreadsNonBlocking<TDev> const& queue) -> bool
{
return queue.m_spQueueImpl->m_workerThread->isIdle();
}
};
} 
} 

#include <alpaka/event/EventGenericThreads.hpp>
