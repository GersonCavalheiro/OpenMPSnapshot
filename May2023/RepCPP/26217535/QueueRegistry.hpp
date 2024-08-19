

#pragma once

#include <alpaka/core/Common.hpp>

#include <deque>
#include <functional>
#include <memory>
#include <mutex>

namespace alpaka
{
namespace detail
{
template<typename TQueue>
class QueueRegistry
{
public:
ALPAKA_FN_HOST auto getAllExistingQueues() const -> std::vector<std::shared_ptr<TQueue>>
{
std::vector<std::shared_ptr<TQueue>> vspQueues;

std::lock_guard<std::mutex> lk(m_Mutex);
vspQueues.reserve(std::size(m_queues));

for(auto it = std::begin(m_queues); it != std::end(m_queues);)
{
auto spQueue = it->lock();
if(spQueue)
{
vspQueues.emplace_back(std::move(spQueue));
++it;
}
else
{
it = m_queues.erase(it);
}
}
return vspQueues;
}

ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<TQueue> spQueue) const -> void
{
std::lock_guard<std::mutex> lk(m_Mutex);

m_queues.push_back(std::move(spQueue));
}

using CleanerFunctor = std::function<void()>;
static ALPAKA_FN_HOST auto registerCleanup(CleanerFunctor cleaner) -> void
{
class CleanupList
{
std::mutex m_mutex;
std::deque<CleanerFunctor> mutable m_cleanup;

public:
~CleanupList()
{
for(auto& c : m_cleanup)
{
c();
}
}

void push(CleanerFunctor&& c)
{
std::lock_guard<std::mutex> lk(m_mutex);

m_cleanup.emplace_back(std::move(c));
}
};
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors" 
#endif
static CleanupList cleanupList;
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

cleanupList.push(std::move(cleaner));
}

private:
std::mutex mutable m_Mutex;
std::deque<std::weak_ptr<TQueue>> mutable m_queues;
};
} 
} 
