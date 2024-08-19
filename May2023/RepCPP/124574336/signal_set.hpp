
#ifndef BOOST_ASIO_SIGNAL_SET_HPP
#define BOOST_ASIO_SIGNAL_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/basic_signal_set.hpp>

namespace boost {
namespace asio {

typedef basic_signal_set<> signal_set;

} 
} 

#endif 
