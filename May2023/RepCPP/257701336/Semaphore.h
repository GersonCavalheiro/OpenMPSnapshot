

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <mutex>
#include <condition_variable>

namespace Aws
{
namespace Utils
{
namespace Threading
{
class AWS_CORE_API Semaphore {
public:

Semaphore(size_t initialCount, size_t maxCount);

void WaitOne();

void Release();

void ReleaseAll();
private:
size_t m_count;
const size_t m_maxCount;
std::mutex m_mutex;
std::condition_variable m_syncPoint;
};
}
}
}
