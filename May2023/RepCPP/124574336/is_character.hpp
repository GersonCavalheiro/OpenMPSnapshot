
#ifndef BOOST_LEXICAL_CAST_DETAIL_IS_CHARACTER_HPP
#define BOOST_LEXICAL_CAST_DETAIL_IS_CHARACTER_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost {

namespace detail 
{
template < typename T >
struct is_character
{
typedef BOOST_DEDUCED_TYPENAME boost::integral_constant<
bool,
boost::is_same< T, char >::value ||
#if !defined(BOOST_NO_STRINGSTREAM) && !defined(BOOST_NO_STD_WSTRING)
boost::is_same< T, wchar_t >::value ||
#endif
#ifndef BOOST_NO_CXX11_CHAR16_T
boost::is_same< T, char16_t >::value ||
#endif
#ifndef BOOST_NO_CXX11_CHAR32_T
boost::is_same< T, char32_t >::value ||
#endif
boost::is_same< T, unsigned char >::value ||
boost::is_same< T, signed char >::value
> type;

BOOST_STATIC_CONSTANT(bool, value = (type::value) );
};
}
}

#endif 

