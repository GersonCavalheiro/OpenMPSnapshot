
#ifndef ASIO_MULTIPLE_EXCEPTIONS_HPP
#define ASIO_MULTIPLE_EXCEPTIONS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <exception>
#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_STD_EXCEPTION_PTR) \
|| defined(GENERATING_DOCUMENTATION)

class multiple_exceptions
: public std::exception
{
public:
ASIO_DECL multiple_exceptions(
std::exception_ptr first) ASIO_NOEXCEPT;

ASIO_DECL virtual const char* what() const
ASIO_NOEXCEPT_OR_NOTHROW;

ASIO_DECL std::exception_ptr first_exception() const;

private:
std::exception_ptr first_;
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/impl/multiple_exceptions.ipp"
#endif 

#endif 
