
#ifndef ASIO_DETAIL_FUNCTIONAL_HPP
#define ASIO_DETAIL_FUNCTIONAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include <functional>

#if !defined(ASIO_HAS_STD_FUNCTION)
# include <boost/function.hpp>
#endif 

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_FUNCTION)
using std::function;
#else 
using boost::function;
#endif 

} 

#if defined(ASIO_HAS_STD_REFERENCE_WRAPPER)
using std::ref;
using std::reference_wrapper;
#endif 

} 

#endif 
