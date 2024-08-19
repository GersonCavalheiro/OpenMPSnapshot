
#ifndef BOOST_XPRESSIVE_DETAIL_UTILITY_LITERALS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_UTILITY_LITERALS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp> 
#include <boost/cstdint.hpp> 
#include <boost/detail/workaround.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Char, boost::intmax_t Ch, boost::intmax_t Wch>
struct char_literal;

template<typename Char, boost::intmax_t Ch>
struct char_literal<Char, Ch, Ch>
{
BOOST_STATIC_CONSTANT(boost::intmax_t, value = Ch);
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<typename Char, boost::intmax_t Ch>
boost::intmax_t const char_literal<Char, Ch, Ch>::value;
#endif

template<typename Ch>
struct string_literal;

template<>
struct string_literal<char>
{
static BOOST_CONSTEXPR char const *pick(char const *cstr, wchar_t const *)
{
return cstr;
}

static BOOST_CONSTEXPR char pick(char ch, wchar_t)
{
return ch;
}
};

template<>
struct string_literal<wchar_t>
{
static BOOST_CONSTEXPR wchar_t const *pick(char const *, wchar_t const *cstr)
{
return cstr;
}

static BOOST_CONSTEXPR wchar_t pick(char, wchar_t ch)
{
return ch;
}
};

#if BOOST_WORKAROUND(__MWERKS__, BOOST_TESTED_AT(0x3206))

# define BOOST_XPR_CHAR_(Char, ch) ch
# define BOOST_XPR_CSTR_(Char, st) boost::xpressive::detail::string_literal<Char>::pick(st, L##st)

#else

# define BOOST_XPR_CHAR_(Char, ch) boost::xpressive::detail::char_literal<Char, ch, L##ch>::value
# define BOOST_XPR_CSTR_(Char, st) boost::xpressive::detail::string_literal<Char>::pick(st, L##st)

#endif

}}} 

#endif
