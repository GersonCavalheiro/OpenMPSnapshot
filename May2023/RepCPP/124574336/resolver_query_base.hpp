
#ifndef BOOST_ASIO_IP_RESOLVER_QUERY_BASE_HPP
#define BOOST_ASIO_IP_RESOLVER_QUERY_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/ip/resolver_base.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

class resolver_query_base : public resolver_base
{
protected:
~resolver_query_base()
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
