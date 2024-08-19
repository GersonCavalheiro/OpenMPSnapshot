


#ifndef CATCH_OPTIMIZER_HPP_INCLUDED
#define CATCH_OPTIMIZER_HPP_INCLUDED

#if defined(_MSC_VER)
#   include <atomic> 
#endif

#include <catch2/internal/catch_move_and_forward.hpp>

#include <type_traits>

namespace Catch {
namespace Benchmark {
#if defined(__GNUC__) || defined(__clang__)
template <typename T>
inline void keep_memory(T* p) {
asm volatile("" : : "g"(p) : "memory");
}
inline void keep_memory() {
asm volatile("" : : : "memory");
}

namespace Detail {
inline void optimizer_barrier() { keep_memory(); }
} 
#elif defined(_MSC_VER)

#pragma optimize("", off)
template <typename T>
inline void keep_memory(T* p) {
*reinterpret_cast<char volatile*>(p) = *reinterpret_cast<char const volatile*>(p);
}
#pragma optimize("", on)

namespace Detail {
inline void optimizer_barrier() {
std::atomic_thread_fence(std::memory_order_seq_cst);
}
} 

#endif

template <typename T>
inline void deoptimize_value(T&& x) {
keep_memory(&x);
}

template <typename Fn, typename... Args>
inline auto invoke_deoptimized(Fn&& fn, Args&&... args) -> std::enable_if_t<!std::is_same<void, decltype(fn(args...))>::value> {
deoptimize_value(CATCH_FORWARD(fn) (CATCH_FORWARD(args)...));
}

template <typename Fn, typename... Args>
inline auto invoke_deoptimized(Fn&& fn, Args&&... args) -> std::enable_if_t<std::is_same<void, decltype(fn(args...))>::value> {
CATCH_FORWARD(fn) (CATCH_FORWARD(args)...);
}
} 
} 

#endif 
