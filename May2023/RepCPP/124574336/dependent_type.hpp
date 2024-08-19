
#ifndef BOOST_ASIO_DETAIL_DEPENDENT_TYPE_HPP
#define BOOST_ASIO_DETAIL_DEPENDENT_TYPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename DependsOn, typename T>
struct dependent_type
{
typedef T type;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
