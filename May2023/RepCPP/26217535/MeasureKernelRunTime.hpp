

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/core/DemangleTypeNames.hpp>

#include <type_traits>
#include <utility>

namespace alpaka::test::integ
{
template<typename TCallable>
auto measureRunTimeMs(TCallable&& callable) -> std::chrono::milliseconds::rep
{
auto const start = std::chrono::high_resolution_clock::now();
std::forward<TCallable>(callable)();
auto const end = std::chrono::high_resolution_clock::now();
return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

template<typename TQueue, typename TTask>
auto measureTaskRunTimeMs(TQueue& queue, TTask&& task) -> std::chrono::milliseconds::rep
{
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
std::cout << "measureKernelRunTime("
<< " queue: " << core::demangled<TQueue> << " task: " << core::demangled<std::decay_t<TTask>> << ")"
<< std::endl;
#endif
alpaka::wait(queue);

return measureRunTimeMs(
[&]
{
alpaka::enqueue(queue, std::forward<TTask>(task));

alpaka::wait(queue);
});
}
} 
