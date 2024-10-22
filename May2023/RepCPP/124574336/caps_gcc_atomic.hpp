


#ifndef BOOST_ATOMIC_DETAIL_CAPS_GCC_ATOMIC_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CAPS_GCC_ATOMIC_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/int_sizes.hpp>

#if defined(BOOST_ATOMIC_DETAIL_CORE_ARCH_BACKEND_HEADER)
#include BOOST_ATOMIC_DETAIL_CORE_ARCH_BACKEND_HEADER(boost/atomic/detail/caps_arch_)
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT8_LOCK_FREE __GCC_ATOMIC_CHAR_LOCK_FREE

#if BOOST_ATOMIC_DETAIL_SIZEOF_SHORT == 2
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE __GCC_ATOMIC_SHORT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_INT == 2
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE __GCC_ATOMIC_INT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LONG == 2
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE __GCC_ATOMIC_LONG_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LLONG == 2
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE __GCC_ATOMIC_LLONG_LOCK_FREE
#else
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE 0
#endif

#if BOOST_ATOMIC_DETAIL_SIZEOF_SHORT == 4
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE __GCC_ATOMIC_SHORT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_INT == 4
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE __GCC_ATOMIC_INT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LONG == 4
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE __GCC_ATOMIC_LONG_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LLONG == 4
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE __GCC_ATOMIC_LLONG_LOCK_FREE
#else
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE 0
#endif

#if BOOST_ATOMIC_DETAIL_SIZEOF_SHORT == 8
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE __GCC_ATOMIC_SHORT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_INT == 8
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE __GCC_ATOMIC_INT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LONG == 8
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE __GCC_ATOMIC_LONG_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LLONG == 8
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE __GCC_ATOMIC_LLONG_LOCK_FREE
#else
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE 0
#endif

#if BOOST_ATOMIC_DETAIL_SIZEOF_SHORT == 16
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE __GCC_ATOMIC_SHORT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_INT == 16
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE __GCC_ATOMIC_INT_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LONG == 16
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE __GCC_ATOMIC_LONG_LOCK_FREE
#elif BOOST_ATOMIC_DETAIL_SIZEOF_LLONG == 16
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE __GCC_ATOMIC_LLONG_LOCK_FREE
#else
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE 0
#endif

#if defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG16B) &&\
(\
(defined(BOOST_CLANG) && (__clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ < 5))) ||\
(defined(BOOST_GCC) && BOOST_GCC >= 70000)\
)
#undef BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE 0
#endif

#if defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG8B) && defined(BOOST_CLANG)
#undef BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE 0
#endif

#if !defined(BOOST_ATOMIC_INT128_LOCK_FREE) || (BOOST_ATOMIC_INT128_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE)
#undef BOOST_ATOMIC_INT128_LOCK_FREE
#define BOOST_ATOMIC_INT128_LOCK_FREE BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE
#endif

#if !defined(BOOST_ATOMIC_INT64_LOCK_FREE) || (BOOST_ATOMIC_INT64_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE) || (BOOST_ATOMIC_INT64_LOCK_FREE < BOOST_ATOMIC_INT128_LOCK_FREE)
#undef BOOST_ATOMIC_INT64_LOCK_FREE
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE >= BOOST_ATOMIC_INT128_LOCK_FREE
#define BOOST_ATOMIC_INT64_LOCK_FREE BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE
#else
#define BOOST_ATOMIC_INT64_LOCK_FREE BOOST_ATOMIC_INT128_LOCK_FREE
#endif
#endif

#if !defined(BOOST_ATOMIC_INT32_LOCK_FREE) || (BOOST_ATOMIC_INT32_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE) || (BOOST_ATOMIC_INT32_LOCK_FREE < BOOST_ATOMIC_INT64_LOCK_FREE)
#undef BOOST_ATOMIC_INT32_LOCK_FREE
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE >= BOOST_ATOMIC_INT64_LOCK_FREE
#define BOOST_ATOMIC_INT32_LOCK_FREE BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE
#else
#define BOOST_ATOMIC_INT32_LOCK_FREE BOOST_ATOMIC_INT64_LOCK_FREE
#endif
#endif

#if !defined(BOOST_ATOMIC_INT16_LOCK_FREE) || (BOOST_ATOMIC_INT16_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE) || (BOOST_ATOMIC_INT16_LOCK_FREE < BOOST_ATOMIC_INT32_LOCK_FREE)
#undef BOOST_ATOMIC_INT16_LOCK_FREE
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE >= BOOST_ATOMIC_INT32_LOCK_FREE
#define BOOST_ATOMIC_INT16_LOCK_FREE BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE
#else
#define BOOST_ATOMIC_INT16_LOCK_FREE BOOST_ATOMIC_INT32_LOCK_FREE
#endif
#endif

#if !defined(BOOST_ATOMIC_INT8_LOCK_FREE) || (BOOST_ATOMIC_INT8_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT8_LOCK_FREE) || (BOOST_ATOMIC_INT8_LOCK_FREE < BOOST_ATOMIC_INT16_LOCK_FREE)
#undef BOOST_ATOMIC_INT8_LOCK_FREE
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT8_LOCK_FREE >= BOOST_ATOMIC_INT16_LOCK_FREE
#define BOOST_ATOMIC_INT8_LOCK_FREE BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT8_LOCK_FREE
#else
#define BOOST_ATOMIC_INT8_LOCK_FREE BOOST_ATOMIC_INT16_LOCK_FREE
#endif
#endif

#if !defined(BOOST_ATOMIC_POINTER_LOCK_FREE) || (BOOST_ATOMIC_POINTER_LOCK_FREE < __GCC_ATOMIC_POINTER_LOCK_FREE)
#undef BOOST_ATOMIC_POINTER_LOCK_FREE
#define BOOST_ATOMIC_POINTER_LOCK_FREE __GCC_ATOMIC_POINTER_LOCK_FREE
#endif

#if !defined(BOOST_ATOMIC_THREAD_FENCE) || (BOOST_ATOMIC_THREAD_FENCE < 2)
#undef BOOST_ATOMIC_THREAD_FENCE
#define BOOST_ATOMIC_THREAD_FENCE 2
#endif
#if !defined(BOOST_ATOMIC_SIGNAL_FENCE) || (BOOST_ATOMIC_SIGNAL_FENCE < 2)
#undef BOOST_ATOMIC_SIGNAL_FENCE
#define BOOST_ATOMIC_SIGNAL_FENCE 2
#endif

#endif 
