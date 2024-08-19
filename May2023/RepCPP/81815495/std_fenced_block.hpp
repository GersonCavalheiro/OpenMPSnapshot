
#ifndef ASIO_DETAIL_STD_FENCED_BLOCK_HPP
#define ASIO_DETAIL_STD_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_ATOMIC)

#include <atomic>
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class std_fenced_block
: private noncopyable
{
public:
enum half_t { half };
enum full_t { full };

explicit std_fenced_block(half_t)
{
}

explicit std_fenced_block(full_t)
{
std::atomic_thread_fence(std::memory_order_acquire);
}

~std_fenced_block()
{
std::atomic_thread_fence(std::memory_order_release);
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
