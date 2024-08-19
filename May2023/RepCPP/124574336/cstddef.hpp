
#ifndef BOOST_ASIO_DETAIL_CSTDDEF_HPP
#define BOOST_ASIO_DETAIL_CSTDDEF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>

namespace boost {
namespace asio {

#if defined(BOOST_ASIO_HAS_NULLPTR)
using std::nullptr_t;
#else 
struct nullptr_t {};
#endif 

} 
} 

#endif 
