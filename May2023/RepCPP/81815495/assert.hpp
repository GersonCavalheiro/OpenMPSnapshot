
#ifndef ASIO_DETAIL_ASSERT_HPP
#define ASIO_DETAIL_ASSERT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_ASSERT)
# include <boost/assert.hpp>
#else 
# include <cassert>
#endif 

#if defined(ASIO_HAS_BOOST_ASSERT)
# define ASIO_ASSERT(expr) BOOST_ASSERT(expr)
#else 
# define ASIO_ASSERT(expr) assert(expr)
#endif 

#endif 
