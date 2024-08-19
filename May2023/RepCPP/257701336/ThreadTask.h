

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <functional>
#include <thread>
#include <atomic>

namespace Aws
{
namespace Utils
{
namespace Threading
{
class PooledThreadExecutor;

class AWS_CORE_API ThreadTask
{
public:
ThreadTask(PooledThreadExecutor& executor);
~ThreadTask();


ThreadTask(const ThreadTask&) = delete;
ThreadTask& operator =(const ThreadTask&) = delete;
ThreadTask(ThreadTask&&) = delete;
ThreadTask& operator =(ThreadTask&&) = delete;

void StopProcessingWork();                

protected:
void MainTaskRunner();

private:                
std::atomic<bool> m_continue;
PooledThreadExecutor& m_executor;
std::thread m_thread;
};
}
}
}
