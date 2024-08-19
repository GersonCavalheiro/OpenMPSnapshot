
#ifndef ASIO_IP_BAD_ADDRESS_CAST_HPP
#define ASIO_IP_BAD_ADDRESS_CAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <typeinfo>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

class bad_address_cast :
#if defined(ASIO_MSVC) && defined(_HAS_EXCEPTIONS) && !_HAS_EXCEPTIONS
public std::exception
#else
public std::bad_cast
#endif
{
public:
bad_address_cast() {}

virtual ~bad_address_cast() ASIO_NOEXCEPT_OR_NOTHROW {}

virtual const char* what() const ASIO_NOEXCEPT_OR_NOTHROW
{
return "bad address cast";
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
