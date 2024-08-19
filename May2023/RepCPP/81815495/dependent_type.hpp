
#ifndef ASIO_DETAIL_DEPENDENT_TYPE_HPP
#define ASIO_DETAIL_DEPENDENT_TYPE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename DependsOn, typename T>
struct dependent_type
{
typedef T type;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
