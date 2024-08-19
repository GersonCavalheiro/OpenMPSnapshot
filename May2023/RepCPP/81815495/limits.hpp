
#ifndef ASIO_DETAIL_LIMITS_HPP
#define ASIO_DETAIL_LIMITS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_LIMITS)
# include <boost/limits.hpp>
#else 
# include <limits>
#endif 

#endif 
