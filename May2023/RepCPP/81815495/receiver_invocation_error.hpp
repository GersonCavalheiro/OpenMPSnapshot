
#ifndef ASIO_EXECUTION_RECEIVER_INVOCATION_ERROR_HPP
#define ASIO_EXECUTION_RECEIVER_INVOCATION_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <stdexcept>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

class receiver_invocation_error
: public std::runtime_error
#if defined(ASIO_HAS_STD_NESTED_EXCEPTION)
, public std::nested_exception
#endif 
{
public:
ASIO_DECL receiver_invocation_error();
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/execution/impl/receiver_invocation_error.ipp"
#endif 

#endif 
