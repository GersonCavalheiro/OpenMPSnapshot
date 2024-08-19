
#ifndef ASIO_SIGNAL_SET_HPP
#define ASIO_SIGNAL_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/basic_signal_set.hpp"

namespace asio {

typedef basic_signal_set<> signal_set;

} 

#endif 
