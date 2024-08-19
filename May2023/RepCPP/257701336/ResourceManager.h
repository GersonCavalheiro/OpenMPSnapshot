
#pragma once

#include <aws/core/utils/memory/stl/AWSVector.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cassert>

namespace Aws
{
namespace Utils
{

template< typename RESOURCE_TYPE>
class ExclusiveOwnershipResourceManager
{
public:
ExclusiveOwnershipResourceManager() : m_shutdown(false) {}


RESOURCE_TYPE Acquire()
{
std::unique_lock<std::mutex> locker(m_queueLock);
while(!m_shutdown.load() && m_resources.size() == 0)
{
m_semaphore.wait(locker, [&](){ return m_shutdown.load() || m_resources.size() > 0; });                    
}

assert(!m_shutdown.load());

RESOURCE_TYPE resource = m_resources.back();
m_resources.pop_back();

return resource;
}


bool HasResourcesAvailable()
{
std::lock_guard<std::mutex> locker(m_queueLock);
return m_resources.size() > 0 && !m_shutdown.load();
}


void Release(RESOURCE_TYPE resource)
{
std::unique_lock<std::mutex> locker(m_queueLock);
m_resources.push_back(resource);
locker.unlock();
m_semaphore.notify_one();
}


void PutResource(RESOURCE_TYPE resource)
{
m_resources.push_back(resource);
}


Aws::Vector<RESOURCE_TYPE> ShutdownAndWait(size_t resourceCount)
{
Aws::Vector<RESOURCE_TYPE> resources;
std::unique_lock<std::mutex> locker(m_queueLock);
m_shutdown = true;

while (m_resources.size() < resourceCount)
{
m_semaphore.wait(locker, [&]() { return m_resources.size() == resourceCount; });
}

resources = m_resources;
m_resources.clear();

return resources;
}

private:
Aws::Vector<RESOURCE_TYPE> m_resources;
std::mutex m_queueLock;
std::condition_variable m_semaphore;
std::atomic<bool> m_shutdown;
};
}
}
