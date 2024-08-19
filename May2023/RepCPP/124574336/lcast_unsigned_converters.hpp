
#ifndef BOOST_LEXICAL_CAST_DETAIL_LCAST_UNSIGNED_CONVERTERS_HPP
#define BOOST_LEXICAL_CAST_DETAIL_LCAST_UNSIGNED_CONVERTERS_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <climits>
#include <cstddef>
#include <string>
#include <cstring>
#include <cstdio>
#include <boost/limits.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/static_assert.hpp>
#include <boost/detail/workaround.hpp>


#ifndef BOOST_NO_STD_LOCALE
#   include <locale>
#else
#   ifndef BOOST_LEXICAL_CAST_ASSUME_C_LOCALE
#       error "Unable to use <locale> header. Define BOOST_LEXICAL_CAST_ASSUME_C_LOCALE to force "
#       error "boost::lexical_cast to use only 'C' locale during conversions."
#   endif
#endif

#include <boost/lexical_cast/detail/lcast_char_constants.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/noncopyable.hpp>

namespace boost 
{
namespace detail 
{
template<class T>
inline
BOOST_DEDUCED_TYPENAME boost::make_unsigned<T>::type lcast_to_unsigned(const T value) BOOST_NOEXCEPT {
typedef BOOST_DEDUCED_TYPENAME boost::make_unsigned<T>::type result_type;
return value < 0 
? static_cast<result_type>(0u - static_cast<result_type>(value)) 
: static_cast<result_type>(value);
}
}

namespace detail 
{
template <class Traits, class T, class CharT>
class lcast_put_unsigned: boost::noncopyable {
typedef BOOST_DEDUCED_TYPENAME Traits::int_type int_type;
BOOST_DEDUCED_TYPENAME boost::conditional<
(sizeof(unsigned) > sizeof(T))
, unsigned
, T
>::type         m_value;
CharT*          m_finish;
CharT    const  m_czero;
int_type const  m_zero;

public:
lcast_put_unsigned(const T n_param, CharT* finish) BOOST_NOEXCEPT 
: m_value(n_param), m_finish(finish)
, m_czero(lcast_char_constants<CharT>::zero), m_zero(Traits::to_int_type(m_czero))
{
#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
BOOST_STATIC_ASSERT(!std::numeric_limits<T>::is_signed);
#endif
}

CharT* convert() {
#ifndef BOOST_LEXICAL_CAST_ASSUME_C_LOCALE
std::locale loc;
if (loc == std::locale::classic()) {
return main_convert_loop();
}

typedef std::numpunct<CharT> numpunct;
numpunct const& np = BOOST_USE_FACET(numpunct, loc);
std::string const grouping = np.grouping();
std::string::size_type const grouping_size = grouping.size();

if (!grouping_size || grouping[0] <= 0) {
return main_convert_loop();
}

#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
BOOST_STATIC_ASSERT(std::numeric_limits<T>::digits10 < CHAR_MAX);
#endif
CharT const thousands_sep = np.thousands_sep();
std::string::size_type group = 0; 
char last_grp_size = grouping[0];
char left = last_grp_size;

do {
if (left == 0) {
++group;
if (group < grouping_size) {
char const grp_size = grouping[group];
last_grp_size = (grp_size <= 0 ? static_cast<char>(CHAR_MAX) : grp_size);
}

left = last_grp_size;
--m_finish;
Traits::assign(*m_finish, thousands_sep);
}

--left;
} while (main_convert_iteration());

return m_finish;
#else
return main_convert_loop();
#endif
}

private:
inline bool main_convert_iteration() BOOST_NOEXCEPT {
--m_finish;
int_type const digit = static_cast<int_type>(m_value % 10U);
Traits::assign(*m_finish, Traits::to_char_type(m_zero + digit));
m_value /= 10;
return !!m_value; 
}

inline CharT* main_convert_loop() BOOST_NOEXCEPT {
while (main_convert_iteration());
return m_finish;
}
};
}

namespace detail 
{
template <class Traits, class T, class CharT>
class lcast_ret_unsigned: boost::noncopyable {
bool m_multiplier_overflowed;
T m_multiplier;
T& m_value;
const CharT* const m_begin;
const CharT* m_end;

public:
lcast_ret_unsigned(T& value, const CharT* const begin, const CharT* end) BOOST_NOEXCEPT
: m_multiplier_overflowed(false), m_multiplier(1), m_value(value), m_begin(begin), m_end(end)
{
#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
BOOST_STATIC_ASSERT(!std::numeric_limits<T>::is_signed);

BOOST_STATIC_ASSERT_MSG(std::numeric_limits<T>::is_specialized,
"std::numeric_limits are not specialized for integral type passed to boost::lexical_cast"
);
#endif
}

inline bool convert() {
CharT const czero = lcast_char_constants<CharT>::zero;
--m_end;
m_value = static_cast<T>(0);

if (m_begin > m_end || *m_end < czero || *m_end >= czero + 10)
return false;
m_value = static_cast<T>(*m_end - czero);
--m_end;

#ifdef BOOST_LEXICAL_CAST_ASSUME_C_LOCALE
return main_convert_loop();
#else
std::locale loc;
if (loc == std::locale::classic()) {
return main_convert_loop();
}

typedef std::numpunct<CharT> numpunct;
numpunct const& np = BOOST_USE_FACET(numpunct, loc);
std::string const& grouping = np.grouping();
std::string::size_type const grouping_size = grouping.size();


if (!grouping_size || grouping[0] <= 0) {
return main_convert_loop();
}

unsigned char current_grouping = 0;
CharT const thousands_sep = np.thousands_sep();
char remained = static_cast<char>(grouping[current_grouping] - 1);

for (;m_end >= m_begin; --m_end)
{
if (remained) {
if (!main_convert_iteration()) {
return false;
}
--remained;
} else {
if ( !Traits::eq(*m_end, thousands_sep) ) 
{

return main_convert_loop();
} else {
if (m_begin == m_end) return false;
if (current_grouping < grouping_size - 1) ++current_grouping;
remained = grouping[current_grouping];
}
}
} 

return true;
#endif
}

private:
inline bool main_convert_iteration() BOOST_NOEXCEPT {
CharT const czero = lcast_char_constants<CharT>::zero;
T const maxv = (std::numeric_limits<T>::max)();

m_multiplier_overflowed = m_multiplier_overflowed || (maxv/10 < m_multiplier);
m_multiplier = static_cast<T>(m_multiplier * 10);

T const dig_value = static_cast<T>(*m_end - czero);
T const new_sub_value = static_cast<T>(m_multiplier * dig_value);

if (*m_end < czero || *m_end >= czero + 10  
|| (dig_value && (                      
m_multiplier_overflowed                             
|| static_cast<T>(maxv / dig_value) < m_multiplier  
|| static_cast<T>(maxv - new_sub_value) < m_value   
))
) return false;

m_value = static_cast<T>(m_value + new_sub_value);

return true;
}

bool main_convert_loop() BOOST_NOEXCEPT {
for ( ; m_end >= m_begin; --m_end) {
if (!main_convert_iteration()) {
return false;
}
}

return true;
}
};
}
} 

#endif 

