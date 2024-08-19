

#pragma once

#include <alpaka/core/Common.hpp>

#include <thread>
#include <utility>

namespace alpaka
{
namespace trait
{
template<typename TThread, typename TSfinae = void>
struct IsThisThread;
} 

template<typename TThread>
ALPAKA_FN_HOST auto isThisThread(TThread const& thread) -> bool
{
return trait::IsThisThread<TThread>::isThisThread(thread);
}

namespace trait
{
template<>
struct IsThisThread<std::thread>
{
ALPAKA_FN_HOST static auto isThisThread(std::thread const& thread) -> bool
{
return std::this_thread::get_id() == thread.get_id();
}
};
} 
} 
