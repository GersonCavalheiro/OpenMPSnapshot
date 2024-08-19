
#ifndef ASIO_DETAIL_ATOMIC_COUNT_HPP
#define ASIO_DETAIL_ATOMIC_COUNT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)
#elif defined(ASIO_HAS_STD_ATOMIC)
# include <atomic>
#else 
# include <boost/detail/atomic_count.hpp>
#endif 

namespace asio {
namespace detail {

#if !defined(ASIO_HAS_THREADS)
typedef long atomic_count;
inline void increment(atomic_count& a, long b) { a += b; }
inline void ref_count_up(atomic_count& a) { ++a; }
inline bool ref_count_down(atomic_count& a) { return --a == 0; }
#elif defined(ASIO_HAS_STD_ATOMIC)
typedef std::atomic<long> atomic_count;
inline void increment(atomic_count& a, long b) { a += b; }

inline void ref_count_up(atomic_count& a)
{
a.fetch_add(1, std::memory_order_relaxed);
}

inline bool ref_count_down(atomic_count& a)
{
if (a.fetch_sub(1, std::memory_order_release) == 1)
{
std::atomic_thread_fence(std::memory_order_acquire);
return true;
}
return false;
}
#else 
typedef boost::detail::atomic_count atomic_count;
inline void increment(atomic_count& a, long b) { while (b > 0) ++a, --b; }
inline void ref_count_up(atomic_count& a) { ++a; }
inline bool ref_count_down(atomic_count& a) { return --a == 0; }
#endif 

} 
} 

#endif 
