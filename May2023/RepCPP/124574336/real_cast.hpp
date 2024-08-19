
#ifndef BOOST_MATH_TOOLS_REAL_CAST_HPP
#define BOOST_MATH_TOOLS_REAL_CAST_HPP

#include <boost/math/tools/config.hpp>

#ifdef _MSC_VER
#pragma once
#endif

namespace boost{ namespace math
{
namespace tools
{
template <class To, class T>
inline BOOST_MATH_CONSTEXPR To real_cast(T t) BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(T) && BOOST_MATH_IS_FLOAT(To))
{
return static_cast<To>(t);
}
} 
} 
} 

#endif 



