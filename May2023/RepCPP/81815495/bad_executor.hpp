
#ifndef ASIO_EXECUTION_BAD_EXECUTOR_HPP
#define ASIO_EXECUTION_BAD_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <exception>
#include "asio/detail/push_options.hpp"

namespace asio {
namespace execution {

class bad_executor
: public std::exception
{
public:
ASIO_DECL bad_executor() ASIO_NOEXCEPT;

ASIO_DECL virtual const char* what() const
ASIO_NOEXCEPT_OR_NOTHROW;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/execution/impl/bad_executor.ipp"
#endif 

#endif 
