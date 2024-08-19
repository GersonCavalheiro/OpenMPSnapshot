
#ifndef BOOST_ASIO_MULTIPLE_EXCEPTIONS_HPP
#define BOOST_ASIO_MULTIPLE_EXCEPTIONS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <exception>
#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if defined(BOOST_ASIO_HAS_STD_EXCEPTION_PTR) \
|| defined(GENERATING_DOCUMENTATION)

class multiple_exceptions
: public std::exception
{
public:
BOOST_ASIO_DECL multiple_exceptions(
std::exception_ptr first) BOOST_ASIO_NOEXCEPT;

BOOST_ASIO_DECL virtual const char* what() const
BOOST_ASIO_NOEXCEPT_OR_NOTHROW;

BOOST_ASIO_DECL std::exception_ptr first_exception() const;

private:
std::exception_ptr first_;
};

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/impl/multiple_exceptions.ipp>
#endif 

#endif 
