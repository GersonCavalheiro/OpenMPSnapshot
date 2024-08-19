
#ifndef BOOST_ASIO_IP_BAD_ADDRESS_CAST_HPP
#define BOOST_ASIO_IP_BAD_ADDRESS_CAST_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <typeinfo>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

class bad_address_cast :
#if defined(BOOST_ASIO_MSVC) && defined(_HAS_EXCEPTIONS) && !_HAS_EXCEPTIONS
public std::exception
#else
public std::bad_cast
#endif
{
public:
bad_address_cast() {}

virtual ~bad_address_cast() BOOST_ASIO_NOEXCEPT_OR_NOTHROW {}

virtual const char* what() const BOOST_ASIO_NOEXCEPT_OR_NOTHROW
{
return "bad address cast";
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
