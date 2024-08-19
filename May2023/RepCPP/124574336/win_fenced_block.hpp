
#ifndef BOOST_ASIO_DETAIL_WIN_FENCED_BLOCK_HPP
#define BOOST_ASIO_DETAIL_WIN_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS) && !defined(UNDER_CE)

#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
#elif defined(BOOST_ASIO_MSVC) \
&& ((BOOST_ASIO_MSVC < 1400) || !defined(MemoryBarrier))
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
#elif defined(BOOST_ASIO_MSVC) \
&& ((BOOST_ASIO_MSVC < 1400) || !defined(MemoryBarrier))
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
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
