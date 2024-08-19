
#ifndef ASIO_DETAIL_WIN_FENCED_BLOCK_HPP
#define ASIO_DETAIL_WIN_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS) && !defined(UNDER_CE)

#include "asio/detail/socket_types.hpp"
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_fenced_block
: private noncopyable
{
public:
enum half_t { half };
enum full_t { full };

explicit win_fenced_block(half_t)
{
}

explicit win_fenced_block(full_t)
{
#if defined(__BORLANDC__)
LONG barrier = 0;
::InterlockedExchange(&barrier, 1);
#elif defined(ASIO_MSVC) \
&& ((ASIO_MSVC < 1400) || !defined(MemoryBarrier))
# if defined(_M_IX86)
#  pragma warning(push)
#  pragma warning(disable:4793)
LONG barrier;
__asm { xchg barrier, eax }
#  pragma warning(pop)
# endif 
#else
MemoryBarrier();
#endif
}

~win_fenced_block()
{
#if defined(__BORLANDC__)
LONG barrier = 0;
::InterlockedExchange(&barrier, 1);
#elif defined(ASIO_MSVC) \
&& ((ASIO_MSVC < 1400) || !defined(MemoryBarrier))
# if defined(_M_IX86)
#  pragma warning(push)
#  pragma warning(disable:4793)
LONG barrier;
__asm { xchg barrier, eax }
#  pragma warning(pop)
# endif 
#else
MemoryBarrier();
#endif
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
