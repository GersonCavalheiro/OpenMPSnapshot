
#ifndef BOOST_ASIO_DETAIL_WIN_IOCP_THREAD_INFO_HPP
#define BOOST_ASIO_DETAIL_WIN_IOCP_THREAD_INFO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/thread_info_base.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

struct win_iocp_thread_info : public thread_info_base
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
