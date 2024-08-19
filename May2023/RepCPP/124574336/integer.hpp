



#ifndef BOOST_INTEGER_HPP
#define BOOST_INTEGER_HPP

#include <boost/integer_fwd.hpp>  

#include <boost/integer_traits.hpp>  
#include <boost/limits.hpp>          
#include <boost/cstdint.hpp>         
#include <boost/static_assert.hpp>

#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

namespace boost
{


template< typename LeastInt >
struct int_fast_t
{
typedef LeastInt fast;
typedef fast     type;
}; 

namespace detail{

template< int Category > struct int_least_helper {}; 
template< int Category > struct uint_least_helper {}; 

#ifdef BOOST_HAS_LONG_LONG
template<> struct int_least_helper<1> { typedef boost::long_long_type least; };
#elif defined(BOOST_HAS_MS_INT64)
template<> struct int_least_helper<1> { typedef __int64 least; };
#endif
template<> struct int_least_helper<2> { typedef long least; };
template<> struct int_least_helper<3> { typedef int least; };
template<> struct int_least_helper<4> { typedef short least; };
template<> struct int_least_helper<5> { typedef signed char least; };
#ifdef BOOST_HAS_LONG_LONG
template<> struct uint_least_helper<1> { typedef boost::ulong_long_type least; };
#elif defined(BOOST_HAS_MS_INT64)
template<> struct uint_least_helper<1> { typedef unsigned __int64 least; };
#endif
template<> struct uint_least_helper<2> { typedef unsigned long least; };
template<> struct uint_least_helper<3> { typedef unsigned int least; };
template<> struct uint_least_helper<4> { typedef unsigned short least; };
template<> struct uint_least_helper<5> { typedef unsigned char least; };

template <int Bits>
struct exact_signed_base_helper{};
template <int Bits>
struct exact_unsigned_base_helper{};

template <> struct exact_signed_base_helper<sizeof(signed char)* CHAR_BIT> { typedef signed char exact; };
template <> struct exact_unsigned_base_helper<sizeof(unsigned char)* CHAR_BIT> { typedef unsigned char exact; };
#if USHRT_MAX != UCHAR_MAX
template <> struct exact_signed_base_helper<sizeof(short)* CHAR_BIT> { typedef short exact; };
template <> struct exact_unsigned_base_helper<sizeof(unsigned short)* CHAR_BIT> { typedef unsigned short exact; };
#endif
#if UINT_MAX != USHRT_MAX
template <> struct exact_signed_base_helper<sizeof(int)* CHAR_BIT> { typedef int exact; };
template <> struct exact_unsigned_base_helper<sizeof(unsigned int)* CHAR_BIT> { typedef unsigned int exact; };
#endif
#if ULONG_MAX != UINT_MAX && ( !defined __TI_COMPILER_VERSION__ || \
( __TI_COMPILER_VERSION__ >= 7000000 && !defined __TI_40BIT_LONG__ ) )
template <> struct exact_signed_base_helper<sizeof(long)* CHAR_BIT> { typedef long exact; };
template <> struct exact_unsigned_base_helper<sizeof(unsigned long)* CHAR_BIT> { typedef unsigned long exact; };
#endif
#if defined(BOOST_HAS_LONG_LONG) &&\
((defined(ULLONG_MAX) && (ULLONG_MAX != ULONG_MAX)) ||\
(defined(ULONG_LONG_MAX) && (ULONG_LONG_MAX != ULONG_MAX)) ||\
(defined(ULONGLONG_MAX) && (ULONGLONG_MAX != ULONG_MAX)) ||\
(defined(_ULLONG_MAX) && (_ULLONG_MAX != ULONG_MAX)))
template <> struct exact_signed_base_helper<sizeof(boost::long_long_type)* CHAR_BIT> { typedef boost::long_long_type exact; };
template <> struct exact_unsigned_base_helper<sizeof(boost::ulong_long_type)* CHAR_BIT> { typedef boost::ulong_long_type exact; };
#endif


} 


template< int Bits >   
struct int_t : public boost::detail::exact_signed_base_helper<Bits>
{
BOOST_STATIC_ASSERT_MSG(Bits <= (int)(sizeof(boost::intmax_t) * CHAR_BIT),
"No suitable signed integer type with the requested number of bits is available.");
typedef typename boost::detail::int_least_helper
<
#ifdef BOOST_HAS_LONG_LONG
(Bits <= (int)(sizeof(boost::long_long_type) * CHAR_BIT)) +
#else
1 +
#endif
(Bits-1 <= ::std::numeric_limits<long>::digits) +
(Bits-1 <= ::std::numeric_limits<int>::digits) +
(Bits-1 <= ::std::numeric_limits<short>::digits) +
(Bits-1 <= ::std::numeric_limits<signed char>::digits)
>::least  least;
typedef typename int_fast_t<least>::type  fast;
};

template< int Bits >   
struct uint_t : public boost::detail::exact_unsigned_base_helper<Bits>
{
BOOST_STATIC_ASSERT_MSG(Bits <= (int)(sizeof(boost::uintmax_t) * CHAR_BIT),
"No suitable unsigned integer type with the requested number of bits is available.");
#if (defined(BOOST_BORLANDC) || defined(__CODEGEAR__)) && defined(BOOST_NO_INTEGRAL_INT64_T)
BOOST_STATIC_CONSTANT(int, s =
6 +
(Bits <= ::std::numeric_limits<unsigned long>::digits) +
(Bits <= ::std::numeric_limits<unsigned int>::digits) +
(Bits <= ::std::numeric_limits<unsigned short>::digits) +
(Bits <= ::std::numeric_limits<unsigned char>::digits));
typedef typename detail::int_least_helper< ::boost::uint_t<Bits>::s>::least least;
#else
typedef typename boost::detail::uint_least_helper
<
#ifdef BOOST_HAS_LONG_LONG
(Bits <= (int)(sizeof(boost::long_long_type) * CHAR_BIT)) +
#else
1 +
#endif
(Bits <= ::std::numeric_limits<unsigned long>::digits) +
(Bits <= ::std::numeric_limits<unsigned int>::digits) +
(Bits <= ::std::numeric_limits<unsigned short>::digits) +
(Bits <= ::std::numeric_limits<unsigned char>::digits)
>::least  least;
#endif
typedef typename int_fast_t<least>::type  fast;
};


#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
template< boost::long_long_type MaxValue >   
#else
template< long MaxValue >   
#endif
struct int_max_value_t
{
typedef typename boost::detail::int_least_helper
<
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
(MaxValue <= ::boost::integer_traits<boost::long_long_type>::const_max) +
#else
1 +
#endif
(MaxValue <= ::boost::integer_traits<long>::const_max) +
(MaxValue <= ::boost::integer_traits<int>::const_max) +
(MaxValue <= ::boost::integer_traits<short>::const_max) +
(MaxValue <= ::boost::integer_traits<signed char>::const_max)
>::least  least;
typedef typename int_fast_t<least>::type  fast;
};

#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
template< boost::long_long_type MinValue >   
#else
template< long MinValue >   
#endif
struct int_min_value_t
{
typedef typename boost::detail::int_least_helper
<
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && !defined(BOOST_NO_INT64_T) && defined(BOOST_HAS_LONG_LONG)
(MinValue >= ::boost::integer_traits<boost::long_long_type>::const_min) +
#else
1 +
#endif
(MinValue >= ::boost::integer_traits<long>::const_min) +
(MinValue >= ::boost::integer_traits<int>::const_min) +
(MinValue >= ::boost::integer_traits<short>::const_min) +
(MinValue >= ::boost::integer_traits<signed char>::const_min)
>::least  least;
typedef typename int_fast_t<least>::type  fast;
};

#if !defined(BOOST_NO_INTEGRAL_INT64_T) && defined(BOOST_HAS_LONG_LONG)
template< boost::ulong_long_type MaxValue >   
#else
template< unsigned long MaxValue >   
#endif
struct uint_value_t
{
#if (defined(BOOST_BORLANDC) || defined(__CODEGEAR__))
#if defined(BOOST_NO_INTEGRAL_INT64_T)
BOOST_STATIC_CONSTANT(unsigned, which =
1 +
(MaxValue <= ::boost::integer_traits<unsigned long>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned int>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned short>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned char>::const_max));
typedef typename detail::int_least_helper< ::boost::uint_value_t<MaxValue>::which>::least least;
#else 
BOOST_STATIC_CONSTANT(unsigned, which =
1 +
(MaxValue <= ::boost::integer_traits<boost::ulong_long_type>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned long>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned int>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned short>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned char>::const_max));
typedef typename detail::uint_least_helper< ::boost::uint_value_t<MaxValue>::which>::least least;
#endif 
#else
typedef typename boost::detail::uint_least_helper
<
#if !defined(BOOST_NO_INTEGRAL_INT64_T) && defined(BOOST_HAS_LONG_LONG)
(MaxValue <= ::boost::integer_traits<boost::ulong_long_type>::const_max) +
#else
1 +
#endif
(MaxValue <= ::boost::integer_traits<unsigned long>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned int>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned short>::const_max) +
(MaxValue <= ::boost::integer_traits<unsigned char>::const_max)
>::least  least;
#endif
typedef typename int_fast_t<least>::type  fast;
};


} 

#endif  
