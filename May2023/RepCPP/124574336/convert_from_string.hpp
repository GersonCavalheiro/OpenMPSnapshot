
#ifndef BOOST_MATH_TOOLS_CONVERT_FROM_STRING_INCLUDED
#define BOOST_MATH_TOOLS_CONVERT_FROM_STRING_INCLUDED

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/type_traits/is_constructible.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/lexical_cast.hpp>

namespace boost{ namespace math{ namespace tools{

template <class T>
struct convert_from_string_result
{
typedef typename boost::conditional<boost::is_constructible<T, const char*>::value, const char*, T>::type type;
};

template <class Real>
Real convert_from_string(const char* p, const boost::false_type&)
{
#ifdef BOOST_MATH_NO_LEXICAL_CAST
BOOST_STATIC_ASSERT(sizeof(Real) == 0);
#else
return boost::lexical_cast<Real>(p);
#endif
}
template <class Real>
BOOST_CONSTEXPR const char* convert_from_string(const char* p, const boost::true_type&) BOOST_NOEXCEPT
{
return p;
}
template <class Real>
BOOST_CONSTEXPR typename convert_from_string_result<Real>::type convert_from_string(const char* p) BOOST_NOEXCEPT_IF((boost::is_constructible<Real, const char*>::value))
{
return convert_from_string<Real>(p, boost::is_constructible<Real, const char*>());
}

} 
} 
} 

#endif 

