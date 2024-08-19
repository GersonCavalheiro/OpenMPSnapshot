
#ifndef ASIO_DETAIL_MACOS_FENCED_BLOCK_HPP
#define ASIO_DETAIL_MACOS_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(__MACH__) && defined(__APPLE__)

#include <libkern/OSAtomic.h>
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class macos_fenced_block
: private noncopyable
{
public:
enum half_t { half };
enum full_t { full };

explicit macos_fenced_block(half_t)
{
}

explicit macos_fenced_block(full_t)
{
OSMemoryBarrier();
}

~macos_fenced_block()
{
OSMemoryBarrier();
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
