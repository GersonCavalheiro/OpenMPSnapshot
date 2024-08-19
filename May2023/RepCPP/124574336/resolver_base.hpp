
#ifndef BOOST_ASIO_IP_RESOLVER_BASE_HPP
#define BOOST_ASIO_IP_RESOLVER_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

class resolver_base
{
public:
#if defined(GENERATING_DOCUMENTATION)
typedef unspecified flags;

static const flags canonical_name = implementation_defined;

static const flags passive = implementation_defined;

static const flags numeric_host = implementation_defined;

static const flags numeric_service = implementation_defined;

static const flags v4_mapped = implementation_defined;

static const flags all_matching = implementation_defined;

static const flags address_configured = implementation_defined;
#else
enum flags
{
canonical_name = BOOST_ASIO_OS_DEF(AI_CANONNAME),
passive = BOOST_ASIO_OS_DEF(AI_PASSIVE),
numeric_host = BOOST_ASIO_OS_DEF(AI_NUMERICHOST),
numeric_service = BOOST_ASIO_OS_DEF(AI_NUMERICSERV),
v4_mapped = BOOST_ASIO_OS_DEF(AI_V4MAPPED),
all_matching = BOOST_ASIO_OS_DEF(AI_ALL),
address_configured = BOOST_ASIO_OS_DEF(AI_ADDRCONFIG)
};


friend flags operator&(flags x, flags y)
{
return static_cast<flags>(
static_cast<unsigned int>(x) & static_cast<unsigned int>(y));
}

friend flags operator|(flags x, flags y)
{
return static_cast<flags>(
static_cast<unsigned int>(x) | static_cast<unsigned int>(y));
}

friend flags operator^(flags x, flags y)
{
return static_cast<flags>(
static_cast<unsigned int>(x) ^ static_cast<unsigned int>(y));
}

friend flags operator~(flags x)
{
return static_cast<flags>(~static_cast<unsigned int>(x));
}

friend flags& operator&=(flags& x, flags y)
{
x = x & y;
return x;
}

friend flags& operator|=(flags& x, flags y)
{
x = x | y;
return x;
}

friend flags& operator^=(flags& x, flags y)
{
x = x ^ y;
return x;
}
#endif

protected:
~resolver_base()
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
