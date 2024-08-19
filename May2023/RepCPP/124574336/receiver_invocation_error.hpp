
#ifndef BOOST_ASIO_EXECUTION_RECEIVER_INVOCATION_ERROR_HPP
#define BOOST_ASIO_EXECUTION_RECEIVER_INVOCATION_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <stdexcept>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace execution {

class receiver_invocation_error
: public std::runtime_error
#if defined(BOOST_ASIO_HAS_STD_NESTED_EXCEPTION)
, public std::nested_exception
#endif 
{
public:
BOOST_ASIO_DECL receiver_invocation_error();
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/execution/impl/receiver_invocation_error.ipp>
#endif 

#endif 
