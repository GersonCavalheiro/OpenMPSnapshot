
#ifndef ASIO_DETAIL_ARRAY_HPP
#define ASIO_DETAIL_ARRAY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_ARRAY)
# include <array>
#else 
# include <boost/array.hpp>
#endif 

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_ARRAY)
using std::array;
#else 
using boost::array;
#endif 

} 
} 

#endif 
