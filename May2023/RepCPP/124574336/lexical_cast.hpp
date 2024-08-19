
#ifndef BOOST_LEXICAL_CAST_INCLUDED
#define BOOST_LEXICAL_CAST_INCLUDED

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#if defined(BOOST_NO_STRINGSTREAM) || defined(BOOST_NO_STD_WSTRING)
#define BOOST_LCAST_NO_WCHAR_T
#endif

#include <boost/range/iterator_range_core.hpp>
#include <boost/lexical_cast/bad_lexical_cast.hpp>
#include <boost/lexical_cast/try_lexical_convert.hpp>

namespace boost 
{
template <typename Target, typename Source>
inline Target lexical_cast(const Source &arg)
{
Target result = Target();

if (!boost::conversion::detail::try_lexical_convert(arg, result)) {
boost::conversion::detail::throw_bad_cast<Source, Target>();
}

return result;
}

template <typename Target>
inline Target lexical_cast(const char* chars, std::size_t count)
{
return ::boost::lexical_cast<Target>(
::boost::iterator_range<const char*>(chars, chars + count)
);
}

template <typename Target>
inline Target lexical_cast(const unsigned char* chars, std::size_t count)
{
return ::boost::lexical_cast<Target>(
::boost::iterator_range<const unsigned char*>(chars, chars + count)
);
}

template <typename Target>
inline Target lexical_cast(const signed char* chars, std::size_t count)
{
return ::boost::lexical_cast<Target>(
::boost::iterator_range<const signed char*>(chars, chars + count)
);
}

#ifndef BOOST_LCAST_NO_WCHAR_T
template <typename Target>
inline Target lexical_cast(const wchar_t* chars, std::size_t count)
{
return ::boost::lexical_cast<Target>(
::boost::iterator_range<const wchar_t*>(chars, chars + count)
);
}
#endif
#ifndef BOOST_NO_CXX11_CHAR16_T
template <typename Target>
inline Target lexical_cast(const char16_t* chars, std::size_t count)
{
return ::boost::lexical_cast<Target>(
::boost::iterator_range<const char16_t*>(chars, chars + count)
);
}
#endif
#ifndef BOOST_NO_CXX11_CHAR32_T
template <typename Target>
inline Target lexical_cast(const char32_t* chars, std::size_t count)
{
return ::boost::lexical_cast<Target>(
::boost::iterator_range<const char32_t*>(chars, chars + count)
);
}
#endif

} 

#undef BOOST_LCAST_NO_WCHAR_T

#endif 

