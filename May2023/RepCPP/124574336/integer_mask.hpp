


#ifndef BOOST_INTEGER_INTEGER_MASK_HPP
#define BOOST_INTEGER_INTEGER_MASK_HPP

#include <boost/integer_fwd.hpp>  

#include <boost/config.hpp>   
#include <boost/integer.hpp>  

#include <climits>  
#include <cstddef>  

#include <boost/limits.hpp>  

#if defined(__GNUC__) && (__GNUC__ >= 4)
#pragma GCC system_header
#endif

namespace boost
{



template < std::size_t Bit >
struct high_bit_mask_t
{
typedef typename uint_t<(Bit + 1)>::least  least;
typedef typename uint_t<(Bit + 1)>::fast   fast;

BOOST_STATIC_CONSTANT( least, high_bit = (least( 1u ) << Bit) );
BOOST_STATIC_CONSTANT( fast, high_bit_fast = (fast( 1u ) << Bit) );

BOOST_STATIC_CONSTANT( std::size_t, bit_position = Bit );

};  



#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4310)  
#endif

template < std::size_t Bits >
struct low_bits_mask_t
{
typedef typename uint_t<Bits>::least  least;
typedef typename uint_t<Bits>::fast   fast;

BOOST_STATIC_CONSTANT( least, sig_bits = least(~(least(~(least( 0u ))) << Bits )) );
BOOST_STATIC_CONSTANT( fast, sig_bits_fast = fast(sig_bits) );

BOOST_STATIC_CONSTANT( std::size_t, bit_count = Bits );

};  

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#define BOOST_LOW_BITS_MASK_SPECIALIZE( Type )                                  \
template <  >  struct low_bits_mask_t< std::numeric_limits<Type>::digits >  { \
typedef std::numeric_limits<Type>           limits_type;                  \
typedef uint_t<limits_type::digits>::least  least;                        \
typedef uint_t<limits_type::digits>::fast   fast;                         \
BOOST_STATIC_CONSTANT( least, sig_bits = (~( least(0u) )) );              \
BOOST_STATIC_CONSTANT( fast, sig_bits_fast = fast(sig_bits) );            \
BOOST_STATIC_CONSTANT( std::size_t, bit_count = limits_type::digits );    \
}

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4245)  
#endif

BOOST_LOW_BITS_MASK_SPECIALIZE( unsigned char );

#if USHRT_MAX > UCHAR_MAX
BOOST_LOW_BITS_MASK_SPECIALIZE( unsigned short );
#endif

#if UINT_MAX > USHRT_MAX
BOOST_LOW_BITS_MASK_SPECIALIZE( unsigned int );
#endif

#if ULONG_MAX > UINT_MAX
BOOST_LOW_BITS_MASK_SPECIALIZE( unsigned long );
#endif

#if defined(BOOST_HAS_LONG_LONG)
#if ((defined(ULLONG_MAX) && (ULLONG_MAX > ULONG_MAX)) ||\
(defined(ULONG_LONG_MAX) && (ULONG_LONG_MAX > ULONG_MAX)) ||\
(defined(ULONGLONG_MAX) && (ULONGLONG_MAX > ULONG_MAX)) ||\
(defined(_ULLONG_MAX) && (_ULLONG_MAX > ULONG_MAX)))
BOOST_LOW_BITS_MASK_SPECIALIZE( boost::ulong_long_type );
#endif
#elif defined(BOOST_HAS_MS_INT64)
#if 18446744073709551615ui64 > ULONG_MAX
BOOST_LOW_BITS_MASK_SPECIALIZE( unsigned __int64 );
#endif
#endif

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#undef BOOST_LOW_BITS_MASK_SPECIALIZE


}  


#endif  
