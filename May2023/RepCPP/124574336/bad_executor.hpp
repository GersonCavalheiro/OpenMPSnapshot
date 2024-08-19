
#ifndef BOOST_ASIO_EXECUTION_BAD_EXECUTOR_HPP
#define BOOST_ASIO_EXECUTION_BAD_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <exception>
#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace execution {

class bad_executor
: public std::exception
{
public:
BOOST_ASIO_DECL bad_executor() BOOST_ASIO_NOEXCEPT;

BOOST_ASIO_DECL virtual const char* what() const
BOOST_ASIO_NOEXCEPT_OR_NOTHROW;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/execution/impl/bad_executor.ipp>
#endif 

#endif 
