
#ifndef ASIO_DETAIL_NULL_SIGNAL_BLOCKER_HPP
#define ASIO_DETAIL_NULL_SIGNAL_BLOCKER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS) \
|| defined(ASIO_WINDOWS) \
|| defined(ASIO_WINDOWS_RUNTIME) \
|| defined(__CYGWIN__) \
|| defined(__SYMBIAN32__)

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_signal_blocker
: private noncopyable
{
public:
null_signal_blocker()
{
}

~null_signal_blocker()
{
}

void block()
{
}

void unblock()
{
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
