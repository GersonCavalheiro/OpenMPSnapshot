
#ifndef ASIO_IP_RESOLVER_QUERY_BASE_HPP
#define ASIO_IP_RESOLVER_QUERY_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/ip/resolver_base.hpp"

#include "asio/detail/push_options.hpp"

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

#include "asio/detail/pop_options.hpp"

#endif 
