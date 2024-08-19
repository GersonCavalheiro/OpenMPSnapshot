

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/memory/stl/AWSFunction.h>
#include <aws/core/utils/memory/stl/AWSQueue.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/threading/Semaphore.h>
#include <functional>
#include <future>
#include <mutex>
#include <atomic>

namespace Aws
{
namespace Utils
{
namespace Threading
{
class ThreadTask;


class AWS_CORE_API Executor
{
public:                
virtual ~Executor() = default;


template<class Fn, class ... Args>
bool Submit(Fn&& fn, Args&& ... args)
{
std::function<void()> callable{ std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...) };
return SubmitToThread(std::move(callable));
}

protected:

virtual bool SubmitToThread(std::function<void()>&&) = 0;
};



class AWS_CORE_API DefaultExecutor : public Executor
{
public:
DefaultExecutor() : m_state(State::Free) {}
~DefaultExecutor();
protected:
enum class State
{
Free, Locked, Shutdown
};
bool SubmitToThread(std::function<void()>&&) override;
void Detach(std::thread::id id);
std::atomic<State> m_state;
Aws::UnorderedMap<std::thread::id, std::thread> m_threads;
};

enum class OverflowPolicy
{
QUEUE_TASKS_EVENLY_ACCROSS_THREADS,
REJECT_IMMEDIATELY
};


class AWS_CORE_API PooledThreadExecutor : public Executor
{
public:
PooledThreadExecutor(size_t poolSize, OverflowPolicy overflowPolicy = OverflowPolicy::QUEUE_TASKS_EVENLY_ACCROSS_THREADS);
~PooledThreadExecutor();


PooledThreadExecutor(const PooledThreadExecutor&) = delete;
PooledThreadExecutor& operator =(const PooledThreadExecutor&) = delete;
PooledThreadExecutor(PooledThreadExecutor&&) = delete;
PooledThreadExecutor& operator =(PooledThreadExecutor&&) = delete;

protected:
bool SubmitToThread(std::function<void()>&&) override;

private:
Aws::Queue<std::function<void()>*> m_tasks;
std::mutex m_queueLock;
Aws::Utils::Threading::Semaphore m_sync;
Aws::Vector<ThreadTask*> m_threadTaskHandles;
size_t m_poolSize;
OverflowPolicy m_overflowPolicy;


std::function<void()>* PopTask();
bool HasTasks();

friend class ThreadTask;
};


} 
} 
} 
