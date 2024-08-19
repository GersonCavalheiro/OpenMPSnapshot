

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/ThreadTraits.hpp>

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace alpaka::core
{
namespace detail
{
template<typename T>
struct ThreadSafeQueue
{
ThreadSafeQueue() = default;

[[nodiscard]] auto empty() const -> bool
{
return m_queue.empty();
}

void push(T&& t)
{
std::lock_guard<std::mutex> lk(m_mutex);
m_queue.push(std::move(t));
}

auto pop(T& t) -> bool
{
std::lock_guard<std::mutex> lk(m_mutex);

if(m_queue.empty())
return false;
t = std::move(m_queue.front());
m_queue.pop();
return true;
}

private:
std::queue<T> m_queue;
std::mutex m_mutex;
};

#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#endif
struct ITaskPkg
{
virtual ~ITaskPkg() = default;

void runTask() noexcept
{
try
{
run();
}
catch(...)
{
setException(std::current_exception());
}
}

virtual auto setException(std::exception_ptr const& exceptPtr) -> void = 0;

protected:
virtual auto run() -> void = 0;
};
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif

template<template<typename> class TPromise, typename TFnObj>
struct TaskPkg final : ITaskPkg
{
using TFnObjReturn = decltype(std::declval<TFnObj>()());

TaskPkg(TFnObj&& func) : m_Promise(), m_FnObj(std::move(func))
{
}

void setException(std::exception_ptr const& exceptPtr) final
{
m_Promise.set_exception(exceptPtr);
}
TPromise<TFnObjReturn> m_Promise;

private:
void run() final
{
if constexpr(std::is_void_v<TFnObjReturn>)
{
this->m_FnObj();
m_Promise.set_value();
}
else
m_Promise.set_value(this->m_FnObj());
}

std::remove_reference_t<TFnObj> m_FnObj;
};

template<typename TFnObj0, typename TFnObj1>
auto invokeBothReturnFirst(TFnObj0&& fn0, TFnObj1&& fn1)
{
if constexpr(!std::is_same_v<void, decltype(std::declval<TFnObj0>()())>)
{
auto ret = fn0();
fn1();
return ret;
}
else
{
fn0();
fn1();
}
}

template<typename TMutex, typename TCondVar>
struct ConcurrentExecPoolMutexAndCond
{
TMutex m_mtxWakeup;
TCondVar m_cvWakeup;
};

struct Empty
{
};

template<
typename TIdx,
typename TConcurrentExec,
template<typename TFnObjReturn>
typename TPromise,
typename TYield,
typename TMutex = void,
typename TCondVar = void,
bool TisYielding = true>
struct ConcurrentExecPool final
: std::conditional_t<TisYielding, Empty, ConcurrentExecPoolMutexAndCond<TMutex, TCondVar>>
{
ConcurrentExecPool(TIdx concurrentExecutionCount)
{
if(concurrentExecutionCount < 1)
{
throw std::invalid_argument(
"The argument 'concurrentExecutionCount' has to be greate or equal to one!");
}

m_vConcurrentExecs.reserve(static_cast<std::size_t>(concurrentExecutionCount));

for(TIdx i = 0; i < concurrentExecutionCount; ++i)
m_vConcurrentExecs.emplace_back([this]() { concurrentExecFn(); });
}

ConcurrentExecPool(ConcurrentExecPool const&) = delete;
auto operator=(ConcurrentExecPool const&) -> ConcurrentExecPool& = delete;

~ConcurrentExecPool()
{
if constexpr(TisYielding)
m_bShutdownFlag.store(true);
else
{
{
std::unique_lock<TMutex> lock(this->m_mtxWakeup);
m_bShutdownFlag = true;
}
this->m_cvWakeup.notify_all();
}

joinAllConcurrentExecs();

while(auto task = popTask())
{
auto const except
= std::runtime_error("Could not perform task before ConcurrentExecPool destruction");
task->setException(std::make_exception_ptr(except));
}
}

template<typename TFnObj, typename... TArgs>
auto enqueueTask(TFnObj&& task, TArgs&&... args)
{
auto boundTask = [=]() { return task(args...); };
auto decrementNumActiveTasks = [this]() { --m_numActiveTasks; };

auto extendedTask = [boundTask, decrementNumActiveTasks]()
{ return invokeBothReturnFirst(std::move(boundTask), std::move(decrementNumActiveTasks)); };

using TaskPackage = TaskPkg<TPromise, decltype(extendedTask)>;
auto pTaskPackage = new TaskPackage(std::move(extendedTask));
std::shared_ptr<ITaskPkg> upTaskPackage(pTaskPackage);

auto future = pTaskPackage->m_Promise.get_future();

++m_numActiveTasks;
if constexpr(TisYielding)
m_qTasks.push(std::move(upTaskPackage));
else
{
{
std::lock_guard<TMutex> lock(this->m_mtxWakeup);
m_qTasks.push(std::move(upTaskPackage));
}

this->m_cvWakeup.notify_one();
}

return future;
}

[[nodiscard]] auto getConcurrentExecutionCount() const -> TIdx
{
return std::size(m_vConcurrentExecs);
}

[[nodiscard]] auto isIdle() const -> bool
{
return m_numActiveTasks == 0u;
}

void detach(std::shared_ptr<ConcurrentExecPool>&& self)
{
m_self = std::move(self);
if constexpr(TisYielding)
m_bDetachedFlag = true;
else
{
std::lock_guard<TMutex> lock(this->m_mtxWakeup);
m_bDetachedFlag = true;
this->m_cvWakeup.notify_one();
}
}

auto takeDetachHandle() -> std::shared_ptr<ConcurrentExecPool>
{
if(m_bDetachedFlag.exchange(false))
return std::move(m_self);
else
return nullptr;
}

private:
void concurrentExecFn()
{
while(!m_bShutdownFlag.load(std::memory_order_relaxed))
{
if constexpr(TisYielding)
{
if(auto task = popTask())
task->runTask();
else
{
if(takeDetachHandle())
return; 
TYield::yield();
}
}
else
{
if(auto task = popTask())
task->runTask();

std::unique_lock<TMutex> lock(this->m_mtxWakeup);
if(m_qTasks.empty())
{
auto self = takeDetachHandle();
if(self)
{
lock.unlock(); 
return;
}

if(m_bShutdownFlag)
return;

this->m_cvWakeup.wait(
lock,
[this] { return !m_qTasks.empty() || m_bShutdownFlag || m_bDetachedFlag; });
}
}
}
}

void joinAllConcurrentExecs()
{
for(auto&& concurrentExec : m_vConcurrentExecs)
{
if(isThisThread(concurrentExec))
concurrentExec.detach();
else
concurrentExec.join();
}
}

auto popTask() -> std::shared_ptr<ITaskPkg>
{
std::shared_ptr<ITaskPkg> out;
if(m_qTasks.pop(out))
return out;
else
return nullptr;
}

private:
std::vector<TConcurrentExec> m_vConcurrentExecs;
ThreadSafeQueue<std::shared_ptr<ITaskPkg>> m_qTasks;
std::atomic<std::uint32_t> m_numActiveTasks = 0u;
std::atomic<bool> m_bShutdownFlag = false;
std::atomic<bool> m_bDetachedFlag = false;
std::shared_ptr<ConcurrentExecPool> m_self = nullptr;
};
} 
} 
