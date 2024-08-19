
#ifndef BOOST_ASIO_DETAIL_NULL_SIGNAL_BLOCKER_HPP
#define BOOST_ASIO_DETAIL_NULL_SIGNAL_BLOCKER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_HAS_THREADS) \
|| defined(BOOST_ASIO_WINDOWS) \
|| defined(BOOST_ASIO_WINDOWS_RUNTIME) \
|| defined(__CYGWIN__) \
|| defined(__SYMBIAN32__)

#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
