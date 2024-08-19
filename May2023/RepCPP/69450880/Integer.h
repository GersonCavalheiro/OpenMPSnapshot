





#ifndef INTEGER_H_
#define INTEGER_H_

#include <hydra/detail/Config.h>
#include <type_traits>
#include <limits>
#include <cstdint>
#include <climits>


namespace hydra {

namespace detail
{


template< typename LeastInt >
struct int_fast_t
{
typedef LeastInt fast;
typedef fast     type;
}; 

namespace impl{

template< int Category > struct int_least_helper {}; 
template< int Category > struct uint_least_helper {}; 


template<> struct int_least_helper<1> { typedef long long least; };
template<> struct int_least_helper<2> { typedef long least; };
template<> struct int_least_helper<3> { typedef int least; };
template<> struct int_least_helper<4> { typedef short least; };
template<> struct int_least_helper<5> { typedef signed char least; };

template<> struct uint_least_helper<1> { typedef unsigned long long least; };
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

#if ULONG_LONG_MAX != ULONG_MAX
template <> struct exact_signed_base_helper<sizeof(long long)* CHAR_BIT> { typedef long long exact; };
template <> struct exact_unsigned_base_helper<sizeof(unsigned long long)* CHAR_BIT> { typedef unsigned  long long exact; };
#endif


} 


template< int Bits >   
struct int_t : public impl::exact_signed_base_helper<Bits>
{
static_assert(Bits <= (int)(sizeof(long long) * CHAR_BIT),
"No suitable signed integer type with the requested number of bits is available.");

typedef typename impl::int_least_helper
< (Bits   <= (int)(sizeof(long long) * CHAR_BIT)) +
(Bits-1 <= std::numeric_limits<long>::digits) +
(Bits-1 <= std::numeric_limits<int>::digits)  +
(Bits-1 <= std::numeric_limits<short>::digits) +
(Bits-1 <= std::numeric_limits<signed char>::digits)
>::least  least;

typedef typename int_fast_t<least>::type  fast;
};

template< int Bits >   
struct uint_t : public impl::exact_unsigned_base_helper<Bits>
{
static_assert(Bits <= (int)(sizeof(unsigned long long) * CHAR_BIT),
"No suitable unsigned integer type with the requested number of bits is available.");

typedef typename impl::uint_least_helper
< (Bits <= (int)(sizeof(unsigned long long) * CHAR_BIT)) +
(Bits <= ::std::numeric_limits<unsigned long>::digits) +
(Bits <= ::std::numeric_limits<unsigned int>::digits) +
(Bits <= ::std::numeric_limits<unsigned short>::digits) +
(Bits <= ::std::numeric_limits<unsigned char>::digits)
>::least  least;

typedef typename int_fast_t<least>::type  fast;
};


} 


}  

#endif 
