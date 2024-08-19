#ifndef BOOST_CONTAINER_DETAIL_ADDRESSOF_HPP
#define BOOST_CONTAINER_DETAIL_ADDRESSOF_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <cstddef>

namespace boost {
namespace container {
namespace dtl {

template <typename T>
BOOST_CONTAINER_FORCEINLINE T* addressof(T& obj)
{
return static_cast<T*>(
static_cast<void*>(
const_cast<char*>(
&reinterpret_cast<const volatile char&>(obj)
)));
}

}  
}  
}  

#endif   
