
#ifndef BOOST_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_HPP
#define BOOST_LEXICAL_CAST_DETAIL_CONVERTER_LEXICAL_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#if defined(BOOST_NO_STRINGSTREAM) || defined(BOOST_NO_STD_WSTRING)
#define BOOST_LCAST_NO_WCHAR_T
#endif

#include <cstddef>
#include <string>
#include <boost/limits.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/type_identity.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/has_left_shift.hpp>
#include <boost/type_traits/has_right_shift.hpp>
#include <boost/static_assert.hpp>
#include <boost/detail/lcast_precision.hpp>

#include <boost/lexical_cast/detail/widest_char.hpp>
#include <boost/lexical_cast/detail/is_character.hpp>

#ifndef BOOST_NO_CXX11_HDR_ARRAY
#include <array>
#endif

#include <boost/array.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <boost/container/container_fwd.hpp>

#include <boost/lexical_cast/detail/converter_lexical_streams.hpp>

namespace boost {

namespace detail 
{
template < class Char >
struct normalize_single_byte_char
{
typedef Char type;
};

template <>
struct normalize_single_byte_char< signed char >
{
typedef char type;
};

template <>
struct normalize_single_byte_char< unsigned char >
{
typedef char type;
};
}

namespace detail 
{
template < class T > struct deduce_character_type_later {};
}

namespace detail 
{
template < typename Type >
struct stream_char_common: public boost::conditional<
boost::detail::is_character< Type >::value,
Type,
boost::detail::deduce_character_type_later< Type >
> {};

template < typename Char >
struct stream_char_common< Char* >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< Char* >
> {};

template < typename Char >
struct stream_char_common< const Char* >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< const Char* >
> {};

template < typename Char >
struct stream_char_common< boost::iterator_range< Char* > >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< boost::iterator_range< Char* > >
> {};

template < typename Char >
struct stream_char_common< boost::iterator_range< const Char* > >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< boost::iterator_range< const Char* > >
> {};

template < class Char, class Traits, class Alloc >
struct stream_char_common< std::basic_string< Char, Traits, Alloc > >
{
typedef Char type;
};

template < class Char, class Traits, class Alloc >
struct stream_char_common< boost::container::basic_string< Char, Traits, Alloc > >
{
typedef Char type;
};

template < typename Char, std::size_t N >
struct stream_char_common< boost::array< Char, N > >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< boost::array< Char, N > >
> {};

template < typename Char, std::size_t N >
struct stream_char_common< boost::array< const Char, N > >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< boost::array< const Char, N > >
> {};

#ifndef BOOST_NO_CXX11_HDR_ARRAY
template < typename Char, std::size_t N >
struct stream_char_common< std::array<Char, N > >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< std::array< Char, N > >
> {};

template < typename Char, std::size_t N >
struct stream_char_common< std::array< const Char, N > >: public boost::conditional<
boost::detail::is_character< Char >::value,
Char,
boost::detail::deduce_character_type_later< std::array< const Char, N > >
> {};
#endif

#ifdef BOOST_HAS_INT128
template <> struct stream_char_common< boost::int128_type >: public boost::type_identity< char > {};
template <> struct stream_char_common< boost::uint128_type >: public boost::type_identity< char > {};
#endif

#if !defined(BOOST_LCAST_NO_WCHAR_T) && defined(BOOST_NO_INTRINSIC_WCHAR_T)
template <>
struct stream_char_common< wchar_t >
{
typedef char type;
};
#endif
}

namespace detail 
{
template < class Char >
struct deduce_source_char_impl
{
typedef BOOST_DEDUCED_TYPENAME boost::detail::normalize_single_byte_char< Char >::type type;
};

template < class T >
struct deduce_source_char_impl< deduce_character_type_later< T > >
{
typedef boost::has_left_shift< std::basic_ostream< char >, T > result_t;

#if defined(BOOST_LCAST_NO_WCHAR_T)
BOOST_STATIC_ASSERT_MSG((result_t::value),
"Source type is not std::ostream`able and std::wostream`s are not supported by your STL implementation");
typedef char type;
#else
typedef BOOST_DEDUCED_TYPENAME boost::conditional<
result_t::value, char, wchar_t
>::type type;

BOOST_STATIC_ASSERT_MSG((result_t::value || boost::has_left_shift< std::basic_ostream< type >, T >::value),
"Source type is neither std::ostream`able nor std::wostream`able");
#endif
};
}

namespace detail  
{
template < class Char >
struct deduce_target_char_impl
{
typedef BOOST_DEDUCED_TYPENAME normalize_single_byte_char< Char >::type type;
};

template < class T >
struct deduce_target_char_impl< deduce_character_type_later<T> >
{
typedef boost::has_right_shift<std::basic_istream<char>, T > result_t;

#if defined(BOOST_LCAST_NO_WCHAR_T)
BOOST_STATIC_ASSERT_MSG((result_t::value),
"Target type is not std::istream`able and std::wistream`s are not supported by your STL implementation");
typedef char type;
#else
typedef BOOST_DEDUCED_TYPENAME boost::conditional<
result_t::value, char, wchar_t
>::type type;

BOOST_STATIC_ASSERT_MSG((result_t::value || boost::has_right_shift<std::basic_istream<wchar_t>, T >::value),
"Target type is neither std::istream`able nor std::wistream`able");
#endif
};
}

namespace detail  
{

template < class T >
struct deduce_target_char
{
typedef BOOST_DEDUCED_TYPENAME stream_char_common< T >::type stage1_type;
typedef BOOST_DEDUCED_TYPENAME deduce_target_char_impl< stage1_type >::type stage2_type;

typedef stage2_type type;
};

template < class T >
struct deduce_source_char
{
typedef BOOST_DEDUCED_TYPENAME stream_char_common< T >::type stage1_type;
typedef BOOST_DEDUCED_TYPENAME deduce_source_char_impl< stage1_type >::type stage2_type;

typedef stage2_type type;
};
}

namespace detail 
{
template < class Char, class T >
struct extract_char_traits
: boost::false_type
{
typedef std::char_traits< Char > trait_t;
};

template < class Char, class Traits, class Alloc >
struct extract_char_traits< Char, std::basic_string< Char, Traits, Alloc > >
: boost::true_type
{
typedef Traits trait_t;
};

template < class Char, class Traits, class Alloc>
struct extract_char_traits< Char, boost::container::basic_string< Char, Traits, Alloc > >
: boost::true_type
{
typedef Traits trait_t;
};
}

namespace detail 
{
template<class T>
struct array_to_pointer_decay
{
typedef T type;
};

template<class T, std::size_t N>
struct array_to_pointer_decay<T[N]>
{
typedef const T * type;
};
}

namespace detail 
{
template< class Source,         
class Enable = void   
>
struct lcast_src_length
{
BOOST_STATIC_CONSTANT(std::size_t, value = 1);
};

template <class Source>
struct lcast_src_length<
Source, BOOST_DEDUCED_TYPENAME boost::enable_if<boost::is_integral<Source> >::type
>
{
#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
BOOST_STATIC_CONSTANT(std::size_t, value =
std::numeric_limits<Source>::is_signed +
std::numeric_limits<Source>::is_specialized + 
std::numeric_limits<Source>::digits10 * 2
);
#else
BOOST_STATIC_CONSTANT(std::size_t, value = 156);
BOOST_STATIC_ASSERT(sizeof(Source) * CHAR_BIT <= 256);
#endif
};

template<class Source>
struct lcast_src_length<
Source, BOOST_DEDUCED_TYPENAME boost::enable_if<boost::is_float<Source> >::type
>
{

#ifndef BOOST_LCAST_NO_COMPILE_TIME_PRECISION
BOOST_STATIC_ASSERT(
std::numeric_limits<Source>::max_exponent10 <=  999999L &&
std::numeric_limits<Source>::min_exponent10 >= -999999L
);

BOOST_STATIC_CONSTANT(std::size_t, value =
5 + lcast_precision<Source>::value + 6
);
#else 
BOOST_STATIC_CONSTANT(std::size_t, value = 156);
#endif 
};
}

namespace detail 
{
template <class Source, class Target>
struct lexical_cast_stream_traits {
typedef BOOST_DEDUCED_TYPENAME boost::detail::array_to_pointer_decay<Source>::type src;
typedef BOOST_DEDUCED_TYPENAME boost::remove_cv<src>::type            no_cv_src;

typedef boost::detail::deduce_source_char<no_cv_src>                           deduce_src_char_metafunc;
typedef BOOST_DEDUCED_TYPENAME deduce_src_char_metafunc::type           src_char_t;
typedef BOOST_DEDUCED_TYPENAME boost::detail::deduce_target_char<Target>::type target_char_t;

typedef BOOST_DEDUCED_TYPENAME boost::detail::widest_char<
target_char_t, src_char_t
>::type char_type;

#if !defined(BOOST_NO_CXX11_CHAR16_T) && defined(BOOST_NO_CXX11_UNICODE_LITERALS)
BOOST_STATIC_ASSERT_MSG(( !boost::is_same<char16_t, src_char_t>::value
&& !boost::is_same<char16_t, target_char_t>::value),
"Your compiler does not have full support for char16_t" );
#endif
#if !defined(BOOST_NO_CXX11_CHAR32_T) && defined(BOOST_NO_CXX11_UNICODE_LITERALS)
BOOST_STATIC_ASSERT_MSG(( !boost::is_same<char32_t, src_char_t>::value
&& !boost::is_same<char32_t, target_char_t>::value),
"Your compiler does not have full support for char32_t" );
#endif

typedef BOOST_DEDUCED_TYPENAME boost::conditional<
boost::detail::extract_char_traits<char_type, Target>::value,
BOOST_DEDUCED_TYPENAME boost::detail::extract_char_traits<char_type, Target>,
BOOST_DEDUCED_TYPENAME boost::detail::extract_char_traits<char_type, no_cv_src>
>::type::trait_t traits;

typedef boost::integral_constant<
bool,
boost::is_same<char, src_char_t>::value &&                                 
(sizeof(char) != sizeof(target_char_t)) &&  
(!(boost::detail::is_character<no_cv_src>::value))
> is_string_widening_required_t;

typedef boost::integral_constant<
bool,
!(boost::is_integral<no_cv_src>::value ||
boost::detail::is_character<
BOOST_DEDUCED_TYPENAME deduce_src_char_metafunc::stage1_type          
>::value                                                           
)
> is_source_input_not_optimized_t;

BOOST_STATIC_CONSTANT(bool, requires_stringbuf =
(is_string_widening_required_t::value || is_source_input_not_optimized_t::value)
);

typedef boost::detail::lcast_src_length<no_cv_src> len_t;
};
}

namespace detail
{
template<typename Target, typename Source>
struct lexical_converter_impl
{
typedef lexical_cast_stream_traits<Source, Target>  stream_trait;

typedef detail::lexical_istream_limited_src<
BOOST_DEDUCED_TYPENAME stream_trait::char_type,
BOOST_DEDUCED_TYPENAME stream_trait::traits,
stream_trait::requires_stringbuf,
stream_trait::len_t::value + 1
> i_interpreter_type;

typedef detail::lexical_ostream_limited_src<
BOOST_DEDUCED_TYPENAME stream_trait::char_type,
BOOST_DEDUCED_TYPENAME stream_trait::traits
> o_interpreter_type;

static inline bool try_convert(const Source& arg, Target& result) {
i_interpreter_type i_interpreter;

if (!(i_interpreter.operator <<(arg)))
return false;

o_interpreter_type out(i_interpreter.cbegin(), i_interpreter.cend());

if(!(out.operator >>(result)))
return false;

return true;
}
};
}

} 

#undef BOOST_LCAST_NO_WCHAR_T

#endif 

