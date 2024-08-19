
#ifndef BOOST_ASIO_DETAIL_LOCAL_FREE_ON_BLOCK_EXIT_HPP
#define BOOST_ASIO_DETAIL_LOCAL_FREE_ON_BLOCK_EXIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS) || defined(__CYGWIN__)
#if !defined(BOOST_ASIO_WINDOWS_APP)

#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class local_free_on_block_exit
: private noncopyable
{
public:
explicit local_free_on_block_exit(void* p)
: p_(p)
{
}

~local_free_on_block_exit()
{
::LocalFree(p_);
}

private:
void* p_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
#endif 

#endif 
