
#ifndef BOOST_ASIO_HANDLER_CONTINUATION_HOOK_HPP
#define BOOST_ASIO_HANDLER_CONTINUATION_HOOK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {


inline bool asio_handler_is_continuation(...)
{
return false;
}

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
