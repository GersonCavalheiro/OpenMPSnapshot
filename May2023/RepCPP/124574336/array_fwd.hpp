
#ifndef BOOST_ASIO_DETAIL_ARRAY_FWD_HPP
#define BOOST_ASIO_DETAIL_ARRAY_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

namespace boost {

template<class T, std::size_t N>
class array;

} 

#if defined(BOOST_ASIO_HAS_STD_ARRAY)
# include <array>
#endif 

#endif 
