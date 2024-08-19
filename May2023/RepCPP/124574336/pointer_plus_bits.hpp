
#ifndef BOOST_INTRUSIVE_POINTER_PLUS_BITS_HPP
#define BOOST_INTRUSIVE_POINTER_PLUS_BITS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/mpl.hpp> 
#include <boost/intrusive/detail/assert.hpp> 

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif


#if defined(BOOST_GCC)
#  if (BOOST_GCC >= 40600)
#     pragma GCC diagnostic push
#     pragma GCC diagnostic ignored "-Wuninitialized"
#     if (BOOST_GCC >= 40700)
#        pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#     endif
#  endif
#endif

namespace boost {
namespace intrusive {

template<class VoidPointer, std::size_t Alignment>
struct max_pointer_plus_bits
{
static const std::size_t value = 0;
};

template<std::size_t Alignment>
struct max_pointer_plus_bits<void*, Alignment>
{
static const std::size_t value = detail::ls_zeros<Alignment>::value;
};

template<class Pointer, std::size_t NumBits>
struct pointer_plus_bits
#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
{}
#endif
;

template<class T, std::size_t NumBits>
struct pointer_plus_bits<T*, NumBits>
{
static const uintptr_t Mask = uintptr_t((uintptr_t(1u) << NumBits) - 1);
typedef T*        pointer;

BOOST_INTRUSIVE_FORCEINLINE static pointer get_pointer(pointer n)
{  return pointer(uintptr_t(n) & uintptr_t(~Mask));  }

BOOST_INTRUSIVE_FORCEINLINE static void set_pointer(pointer &n, pointer p)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT(0 == (uintptr_t(p) & Mask));
n = pointer(uintptr_t(p) | (uintptr_t(n) & Mask));
}

BOOST_INTRUSIVE_FORCEINLINE static std::size_t get_bits(pointer n)
{  return std::size_t(uintptr_t(n) & Mask);  }

BOOST_INTRUSIVE_FORCEINLINE static void set_bits(pointer &n, std::size_t c)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT(uintptr_t(c) <= Mask);
n = pointer(uintptr_t((get_pointer)(n)) | uintptr_t(c));
}
};

} 
} 

#if defined(BOOST_GCC) && (BOOST_GCC >= 40600)
#  pragma GCC diagnostic pop
#endif

#include <boost/intrusive/detail/config_end.hpp>

#endif 
