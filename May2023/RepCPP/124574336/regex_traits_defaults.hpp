



#ifndef BOOST_REGEX_TRAITS_DEFAULTS_HPP_INCLUDED
#define BOOST_REGEX_TRAITS_DEFAULTS_HPP_INCLUDED

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/regex/config.hpp>

#ifndef BOOST_REGEX_SYNTAX_TYPE_HPP
#include <boost/regex/v4/syntax_type.hpp>
#endif
#ifndef BOOST_REGEX_ERROR_TYPE_HPP
#include <boost/regex/v4/error_type.hpp>
#endif
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/utility/enable_if.hpp>

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std{
using ::strlen;
}
#endif

namespace boost{ namespace BOOST_REGEX_DETAIL_NS{


template <class charT>
inline bool is_extended(charT c)
{
typedef typename make_unsigned<charT>::type unsigned_type; 
return (sizeof(charT) > 1) && (static_cast<unsigned_type>(c) >= 256u); 
}
inline bool is_extended(char)
{ return false; }


BOOST_REGEX_DECL const char* BOOST_REGEX_CALL get_default_syntax(regex_constants::syntax_type n);
BOOST_REGEX_DECL const char* BOOST_REGEX_CALL get_default_error_string(regex_constants::error_type n);
BOOST_REGEX_DECL regex_constants::syntax_type BOOST_REGEX_CALL get_default_syntax_type(char c);
BOOST_REGEX_DECL regex_constants::escape_syntax_type BOOST_REGEX_CALL get_default_escape_syntax_type(char c);

BOOST_REGEX_DECL bool BOOST_REGEX_CALL is_combining_implementation(uint_least16_t s);

template <class charT>
inline bool is_combining(charT c)
{
return (c <= static_cast<charT>(0)) ? false : ((c >= static_cast<charT>((std::numeric_limits<uint_least16_t>::max)())) ? false : is_combining_implementation(static_cast<unsigned short>(c)));
}
template <>
inline bool is_combining<char>(char)
{
return false;
}
template <>
inline bool is_combining<signed char>(signed char)
{
return false;
}
template <>
inline bool is_combining<unsigned char>(unsigned char)
{
return false;
}
#if !defined(__hpux) && !defined(__WINSCW__) 
#ifdef _MSC_VER
template<>
inline bool is_combining<wchar_t>(wchar_t c)
{
return is_combining_implementation(static_cast<unsigned short>(c));
}
#elif !defined(__DECCXX) && !defined(__osf__) && !defined(__OSF__) && defined(WCHAR_MIN) && (WCHAR_MIN == 0) && !defined(BOOST_NO_INTRINSIC_WCHAR_T)
#if defined(WCHAR_MAX) && (WCHAR_MAX <= USHRT_MAX)
template<>
inline bool is_combining<wchar_t>(wchar_t c)
{
return is_combining_implementation(static_cast<unsigned short>(c));
}
#else
template<>
inline bool is_combining<wchar_t>(wchar_t c)
{
return (c >= (std::numeric_limits<uint_least16_t>::max)()) ? false : is_combining_implementation(static_cast<unsigned short>(c));
}
#endif
#endif
#endif

template <class charT>
inline bool is_separator(charT c)
{
return BOOST_REGEX_MAKE_BOOL(
(c == static_cast<charT>('\n'))
|| (c == static_cast<charT>('\r'))
|| (c == static_cast<charT>('\f'))
|| (static_cast<boost::uint16_t>(c) == 0x2028u)
|| (static_cast<boost::uint16_t>(c) == 0x2029u)
|| (static_cast<boost::uint16_t>(c) == 0x85u));
}
template <>
inline bool is_separator<char>(char c)
{
return BOOST_REGEX_MAKE_BOOL((c == '\n') || (c == '\r') || (c == '\f'));
}

BOOST_REGEX_DECL std::string BOOST_REGEX_CALL lookup_default_collate_name(const std::string& name);

template <class charT>
struct character_pointer_range
{
const charT* p1;
const charT* p2;

bool operator < (const character_pointer_range& r)const
{
return std::lexicographical_compare(p1, p2, r.p1, r.p2);
}
bool operator == (const character_pointer_range& r)const
{
return ((p2 - p1) == (r.p2 - r.p1)) && BOOST_REGEX_DETAIL_NS::equal(p1, p2, r.p1);
}
};
template <class charT>
int get_default_class_id(const charT* p1, const charT* p2)
{
static const charT data[73] = {
'a', 'l', 'n', 'u', 'm',
'a', 'l', 'p', 'h', 'a',
'b', 'l', 'a', 'n', 'k',
'c', 'n', 't', 'r', 'l',
'd', 'i', 'g', 'i', 't',
'g', 'r', 'a', 'p', 'h',
'l', 'o', 'w', 'e', 'r',
'p', 'r', 'i', 'n', 't',
'p', 'u', 'n', 'c', 't',
's', 'p', 'a', 'c', 'e',
'u', 'n', 'i', 'c', 'o', 'd', 'e',
'u', 'p', 'p', 'e', 'r',
'v',
'w', 'o', 'r', 'd',
'x', 'd', 'i', 'g', 'i', 't',
};

static const character_pointer_range<charT> ranges[21] =
{
{data+0, data+5,}, 
{data+5, data+10,}, 
{data+10, data+15,}, 
{data+15, data+20,}, 
{data+20, data+21,}, 
{data+20, data+25,}, 
{data+25, data+30,}, 
{data+29, data+30,}, 
{data+30, data+31,}, 
{data+30, data+35,}, 
{data+35, data+40,}, 
{data+40, data+45,}, 
{data+45, data+46,}, 
{data+45, data+50,}, 
{data+57, data+58,}, 
{data+50, data+57,}, 
{data+57, data+62,}, 
{data+62, data+63,}, 
{data+63, data+64,}, 
{data+63, data+67,}, 
{data+67, data+73,}, 
};
const character_pointer_range<charT>* ranges_begin = ranges;
const character_pointer_range<charT>* ranges_end = ranges + (sizeof(ranges)/sizeof(ranges[0]));

character_pointer_range<charT> t = { p1, p2, };
const character_pointer_range<charT>* p = std::lower_bound(ranges_begin, ranges_end, t);
if((p != ranges_end) && (t == *p))
return static_cast<int>(p - ranges);
return -1;
}

template <class charT>
std::ptrdiff_t global_length(const charT* p)
{
std::ptrdiff_t n = 0;
while(*p)
{
++p;
++n;
}
return n;
}
template<>
inline std::ptrdiff_t global_length<char>(const char* p)
{
return (std::strlen)(p);
}
#ifndef BOOST_NO_WREGEX
template<>
inline std::ptrdiff_t global_length<wchar_t>(const wchar_t* p)
{
return (std::ptrdiff_t)(std::wcslen)(p);
}
#endif
template <class charT>
inline charT BOOST_REGEX_CALL global_lower(charT c)
{
return c;
}
template <class charT>
inline charT BOOST_REGEX_CALL global_upper(charT c)
{
return c;
}

BOOST_REGEX_DECL char BOOST_REGEX_CALL do_global_lower(char c);
BOOST_REGEX_DECL char BOOST_REGEX_CALL do_global_upper(char c);
#ifndef BOOST_NO_WREGEX
BOOST_REGEX_DECL wchar_t BOOST_REGEX_CALL do_global_lower(wchar_t c);
BOOST_REGEX_DECL wchar_t BOOST_REGEX_CALL do_global_upper(wchar_t c);
#endif
#ifdef BOOST_REGEX_HAS_OTHER_WCHAR_T
BOOST_REGEX_DECL unsigned short BOOST_REGEX_CALL do_global_lower(unsigned short c);
BOOST_REGEX_DECL unsigned short BOOST_REGEX_CALL do_global_upper(unsigned short c);
#endif
template<> inline char BOOST_REGEX_CALL global_lower<char>(char c){ return do_global_lower(c); }
template<> inline char BOOST_REGEX_CALL global_upper<char>(char c){ return do_global_upper(c); }
#ifndef BOOST_NO_WREGEX
template<> inline wchar_t BOOST_REGEX_CALL global_lower<wchar_t>(wchar_t c){ return do_global_lower(c); }
template<> inline wchar_t BOOST_REGEX_CALL global_upper<wchar_t>(wchar_t c){ return do_global_upper(c); }
#endif
#ifdef BOOST_REGEX_HAS_OTHER_WCHAR_T
template<> inline unsigned short BOOST_REGEX_CALL global_lower<unsigned short>(unsigned short c){ return do_global_lower(c); }
template<> inline unsigned short BOOST_REGEX_CALL global_upper<unsigned short>(unsigned short c){ return do_global_upper(c); }
#endif

template <class charT>
int global_value(charT c)
{
static const charT zero = '0';
static const charT nine = '9';
static const charT a = 'a';
static const charT f = 'f';
static const charT A = 'A';
static const charT F = 'F';

if(c > f) return -1;
if(c >= a) return 10 + (c - a);
if(c > F) return -1;
if(c >= A) return 10 + (c - A);
if(c > nine) return -1;
if(c >= zero) return c - zero;
return -1;
}
template <class charT, class traits>
boost::intmax_t global_toi(const charT*& p1, const charT* p2, int radix, const traits& t)
{
(void)t; 
boost::intmax_t limit = (std::numeric_limits<boost::intmax_t>::max)() / radix;
boost::intmax_t next_value = t.value(*p1, radix);
if((p1 == p2) || (next_value < 0) || (next_value >= radix))
return -1;
boost::intmax_t result = 0;
while(p1 != p2)
{
next_value = t.value(*p1, radix);
if((next_value < 0) || (next_value >= radix))
break;
result *= radix;
result += next_value;
++p1;
if (result > limit)
return -1;
}
return result;
}

template <class charT>
inline typename boost::enable_if_c<(sizeof(charT) > 1), const charT*>::type get_escape_R_string()
{
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable:4309 4245)
#endif
static const charT e1[] = { '(', '?', '-', 'x', ':', '(', '?', '>', '\x0D', '\x0A', '?',
'|', '[', '\x0A', '\x0B', '\x0C', static_cast<charT>(0x85), static_cast<charT>(0x2028),
static_cast<charT>(0x2029), ']', ')', ')', '\0' };
static const charT e2[] = { '(', '?', '-', 'x', ':', '(', '?', '>', '\x0D', '\x0A', '?',
'|', '[', '\x0A', '\x0B', '\x0C', static_cast<charT>(0x85), ']', ')', ')', '\0' };

charT c = static_cast<charT>(0x2029u);
bool b = (static_cast<unsigned>(c) == 0x2029u);

return (b ? e1 : e2);
#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif
}

template <class charT>
inline typename boost::disable_if_c<(sizeof(charT) > 1), const charT*>::type get_escape_R_string()
{
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable:4309)
#endif
static const charT e2[] = { '(', '?', '-', 'x', ':', '(', '?', '>', '\x0D', '\x0A', '?',
'|', '[', '\x0A', '\x0B', '\x0C', '\x85', ']', ')', ')', '\0' };
return e2;
#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif
}

} 
} 

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
