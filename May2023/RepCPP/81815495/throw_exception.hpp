
#ifndef ASIO_DETAIL_THROW_EXCEPTION_HPP
#define ASIO_DETAIL_THROW_EXCEPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_THROW_EXCEPTION)
# include <boost/throw_exception.hpp>
#endif 

namespace asio {
namespace detail {

#if defined(ASIO_HAS_BOOST_THROW_EXCEPTION)
using boost::throw_exception;
#else 

template <typename Exception>
void throw_exception(const Exception& e);

# if !defined(ASIO_NO_EXCEPTIONS)
template <typename Exception>
void throw_exception(const Exception& e)
{
throw e;
}
# endif 

#endif 

} 
} 

#endif 
