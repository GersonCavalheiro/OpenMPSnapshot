


#ifndef BOOST_ATOMIC_BIT_OPERATION_TOOLS_HPP_INCLUDED_
#define BOOST_ATOMIC_BIT_OPERATION_TOOLS_HPP_INCLUDED_

#include <boost/predef/architecture/x86.h>

#if BOOST_ARCH_X86

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#if defined(_MSC_VER)
extern "C" unsigned char _BitScanForward(unsigned long* index, unsigned long x);
#if defined(BOOST_MSVC)
#pragma intrinsic(_BitScanForward)
#endif
#endif

namespace boost {
namespace atomics {
namespace detail {

BOOST_FORCEINLINE unsigned int count_trailing_zeros(unsigned int x)
{
#if defined(__GNUC__)
return __builtin_ctz(x);
#elif defined(_MSC_VER)
unsigned long index;
_BitScanForward(&index, x);
return static_cast< unsigned int >(index);
#else
unsigned int index = 0u;
if ((x & 0xFFFF) == 0u)
{
x >>= 16;
index += 16u;
}
if ((x & 0xFF) == 0u)
{
x >>= 8;
index += 8u;
}
if ((x & 0xF) == 0u)
{
x >>= 4;
index += 4u;
}
if ((x & 0x3) == 0u)
{
x >>= 2;
index += 2u;
}
if ((x & 0x1) == 0u)
{
index += 1u;
}
return index;
#endif
}

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 

#endif 
