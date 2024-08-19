
#ifndef BOOST_ASIO_DETAIL_SOURCE_LOCATION_HPP
#define BOOST_ASIO_DETAIL_SOURCE_LOCATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_SOURCE_LOCATION)

#if defined(BOOST_ASIO_HAS_STD_SOURCE_LOCATION)
# include <source_location>
#elif defined(BOOST_ASIO_HAS_STD_EXPERIMENTAL_SOURCE_LOCATION)
# include <experimental/source_location>
#else 
# error BOOST_ASIO_HAS_SOURCE_LOCATION is set \
but no source_location is available
#endif 

namespace boost {
namespace asio {
namespace detail {

#if defined(BOOST_ASIO_HAS_STD_SOURCE_LOCATION)
using std::source_location;
#elif defined(BOOST_ASIO_HAS_STD_EXPERIMENTAL_SOURCE_LOCATION)
using std::experimental::source_location;
#endif 

} 
} 
} 

#endif 

#endif 
