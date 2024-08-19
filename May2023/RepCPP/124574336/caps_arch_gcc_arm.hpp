


#ifndef BOOST_ATOMIC_DETAIL_CAPS_ARCH_GCC_ARM_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CAPS_ARCH_GCC_ARM_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/platform.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(__ARMEL__) || \
(defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || \
(defined(__LITTLE_ENDIAN__) && !defined(__BIG_ENDIAN__)) || \
defined(BOOST_WINDOWS)
#define BOOST_ATOMIC_DETAIL_ARM_LITTLE_ENDIAN
#elif defined(__ARMEB__) || \
defined(__ARM_BIG_ENDIAN) || \
(defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__) || \
(defined(__BIG_ENDIAN__) && !defined(__LITTLE_ENDIAN__))
#define BOOST_ATOMIC_DETAIL_ARM_BIG_ENDIAN
#else
#error "Boost.Atomic: Failed to determine ARM endianness, the target platform is not supported. Please, report to the developers (patches are welcome)."
#endif

#if defined(__GNUC__) && defined(__arm__) && (BOOST_ATOMIC_DETAIL_ARM_ARCH >= 6)

#if BOOST_ATOMIC_DETAIL_ARM_ARCH > 6
#define BOOST_ATOMIC_DETAIL_ARM_HAS_DMB 1
#endif

#if defined(__ARM_FEATURE_LDREX)

#if (__ARM_FEATURE_LDREX & 1)
#define BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXB_STREXB 1
#endif
#if (__ARM_FEATURE_LDREX & 2)
#define BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXH_STREXH 1
#endif
#if (__ARM_FEATURE_LDREX & 8)
#define BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXD_STREXD 1
#endif

#else 

#if !(defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6Z__))

#if (__GNUC__ * 100 + __GNUC_MINOR__) >= 409
#define BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXB_STREXB 1
#define BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXH_STREXH 1
#endif

#if !(((defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6ZK__)) && defined(__thumb__)) || defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7M__))
#define BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXD_STREXD 1
#endif

#endif 

#endif 

#endif 

#define BOOST_ATOMIC_INT8_LOCK_FREE 2
#define BOOST_ATOMIC_INT16_LOCK_FREE 2
#define BOOST_ATOMIC_INT32_LOCK_FREE 2
#if defined(BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXD_STREXD)
#define BOOST_ATOMIC_INT64_LOCK_FREE 2
#endif
#define BOOST_ATOMIC_POINTER_LOCK_FREE 2

#define BOOST_ATOMIC_THREAD_FENCE 2
#define BOOST_ATOMIC_SIGNAL_FENCE 2

#endif 
