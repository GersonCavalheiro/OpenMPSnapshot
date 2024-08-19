





#ifndef INTEGERMASK_H_
#define INTEGERMASK_H_


#include <hydra/detail/Config.h>
#include <hydra/detail/utility/Integer.h>
#include <climits>  
#include <cstddef>  
#include <limits>  


namespace hydra {

namespace detail
{



template < std::size_t Bit >
struct high_bit_mask_t
{
typedef typename uint_t<(Bit + 1)>::least  least;
typedef typename uint_t<(Bit + 1)>::fast   fast;

static const  least high_bit = (least( 1u ) << Bit) ;
static const  fast  high_bit_fast = (fast( 1u ) << Bit) ;

static const  std::size_t bit_position = Bit ;

};  




template < std::size_t Bits >
struct low_bits_mask_t
{
typedef typename uint_t<Bits>::least  least;
typedef typename uint_t<Bits>::fast   fast;

static const least sig_bits = least(~(least(~(least( 0u ))) << Bits )) ;
static const fast sig_bits_fast = fast(sig_bits) ;

static const std::size_t bit_count = Bits ;

};  


#define HYDRA_LOW_BITS_MASK_SPECIALIZE( Type )                                  \
template <  >  struct low_bits_mask_t< std::numeric_limits<Type>::digits >  { \
typedef std::numeric_limits<Type>           limits_type;                  \
typedef uint_t<limits_type::digits>::least  least;                        \
typedef uint_t<limits_type::digits>::fast   fast;                         \
static const least sig_bits = (~( least(0u) )) ;                          \
static const fast sig_bits_fast = fast(sig_bits) ;                        \
static const std::size_t bit_count = limits_type::digits ;                \
}

HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned char );

#if USHRT_MAX > UCHAR_MAX
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned short );
#endif

#if UINT_MAX > USHRT_MAX
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned int );
#endif

#if ULONG_MAX > UINT_MAX
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned long );
#endif

#if (defined(ULLONG_MAX) && (ULLONG_MAX > ULONG_MAX))
HYDRA_LOW_BITS_MASK_SPECIALIZE( unsigned long long );
#endif



#undef HYDRA_LOW_BITS_MASK_SPECIALIZE


}  

}  





#endif 
