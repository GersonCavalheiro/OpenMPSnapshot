
#ifndef BOOST_LEXICAL_CAST_DETAIL_INF_NAN_HPP
#define BOOST_LEXICAL_CAST_DETAIL_INF_NAN_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#if defined(BOOST_NO_STRINGSTREAM) || defined(BOOST_NO_STD_WSTRING)
#define BOOST_LCAST_NO_WCHAR_T
#endif

#include <cstddef>
#include <cstring>
#include <boost/limits.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <boost/lexical_cast/detail/lcast_char_constants.hpp>

namespace boost {
namespace detail
{
template <class CharT>
bool lc_iequal(const CharT* val, const CharT* lcase, const CharT* ucase, unsigned int len) BOOST_NOEXCEPT {
for( unsigned int i=0; i < len; ++i ) {
if ( val[i] != lcase[i] && val[i] != ucase[i] ) return false;
}

return true;
}


template <class CharT, class T>
inline bool parse_inf_nan_impl(const CharT* begin, const CharT* end, T& value
, const CharT* lc_NAN, const CharT* lc_nan
, const CharT* lc_INFINITY, const CharT* lc_infinity
, const CharT opening_brace, const CharT closing_brace) BOOST_NOEXCEPT
{
using namespace std;
if (begin == end) return false;
const CharT minus = lcast_char_constants<CharT>::minus;
const CharT plus = lcast_char_constants<CharT>::plus;
const int inifinity_size = 8; 


bool const has_minus = (*begin == minus);
if (has_minus || *begin == plus) {
++ begin;
}

if (end - begin < 3) return false;
if (lc_iequal(begin, lc_nan, lc_NAN, 3)) {
begin += 3;
if (end != begin) {


if (end - begin < 2) return false; 
-- end;
if (*begin != opening_brace || *end != closing_brace) return false; 
}

if( !has_minus ) value = std::numeric_limits<T>::quiet_NaN();
else value = (boost::math::changesign) (std::numeric_limits<T>::quiet_NaN());
return true;
} else if (
( 
end - begin == 3      
&& lc_iequal(begin, lc_infinity, lc_INFINITY, 3)
)
||
( 
end - begin == inifinity_size
&& lc_iequal(begin, lc_infinity, lc_INFINITY, inifinity_size)
)
)
{
if( !has_minus ) value = std::numeric_limits<T>::infinity();
else value = (boost::math::changesign) (std::numeric_limits<T>::infinity());
return true;
}

return false;
}

template <class CharT, class T>
bool put_inf_nan_impl(CharT* begin, CharT*& end, const T& value
, const CharT* lc_nan
, const CharT* lc_infinity) BOOST_NOEXCEPT
{
using namespace std;
const CharT minus = lcast_char_constants<CharT>::minus;
if ((boost::math::isnan)(value)) {
if ((boost::math::signbit)(value)) {
*begin = minus;
++ begin;
}

memcpy(begin, lc_nan, 3 * sizeof(CharT));
end = begin + 3;
return true;
} else if ((boost::math::isinf)(value)) {
if ((boost::math::signbit)(value)) {
*begin = minus;
++ begin;
}

memcpy(begin, lc_infinity, 3 * sizeof(CharT));
end = begin + 3;
return true;
}

return false;
}


#ifndef BOOST_LCAST_NO_WCHAR_T
template <class T>
bool parse_inf_nan(const wchar_t* begin, const wchar_t* end, T& value) BOOST_NOEXCEPT {
return parse_inf_nan_impl(begin, end, value
, L"NAN", L"nan"
, L"INFINITY", L"infinity"
, L'(', L')');
}

template <class T>
bool put_inf_nan(wchar_t* begin, wchar_t*& end, const T& value) BOOST_NOEXCEPT {
return put_inf_nan_impl(begin, end, value, L"nan", L"infinity");
}

#endif
#if !defined(BOOST_NO_CXX11_CHAR16_T) && !defined(BOOST_NO_CXX11_UNICODE_LITERALS)
template <class T>
bool parse_inf_nan(const char16_t* begin, const char16_t* end, T& value) BOOST_NOEXCEPT {
return parse_inf_nan_impl(begin, end, value
, u"NAN", u"nan"
, u"INFINITY", u"infinity"
, u'(', u')');
}

template <class T>
bool put_inf_nan(char16_t* begin, char16_t*& end, const T& value) BOOST_NOEXCEPT {
return put_inf_nan_impl(begin, end, value, u"nan", u"infinity");
}
#endif
#if !defined(BOOST_NO_CXX11_CHAR32_T) && !defined(BOOST_NO_CXX11_UNICODE_LITERALS)
template <class T>
bool parse_inf_nan(const char32_t* begin, const char32_t* end, T& value) BOOST_NOEXCEPT {
return parse_inf_nan_impl(begin, end, value
, U"NAN", U"nan"
, U"INFINITY", U"infinity"
, U'(', U')');
}

template <class T>
bool put_inf_nan(char32_t* begin, char32_t*& end, const T& value) BOOST_NOEXCEPT {
return put_inf_nan_impl(begin, end, value, U"nan", U"infinity");
}
#endif

template <class CharT, class T>
bool parse_inf_nan(const CharT* begin, const CharT* end, T& value) BOOST_NOEXCEPT {
return parse_inf_nan_impl(begin, end, value
, "NAN", "nan"
, "INFINITY", "infinity"
, '(', ')');
}

template <class CharT, class T>
bool put_inf_nan(CharT* begin, CharT*& end, const T& value) BOOST_NOEXCEPT {
return put_inf_nan_impl(begin, end, value, "nan", "infinity");
}
}
} 

#undef BOOST_LCAST_NO_WCHAR_T

#endif 

