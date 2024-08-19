


#ifndef BOOST_ATOMIC_DETAIL_CAPS_ARCH_GCC_X86_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CAPS_ARCH_GCC_X86_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(__GNUC__)

#if defined(__i386__) &&\
(\
defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8) ||\
defined(__i586__) || defined(__i686__) || defined(__SSE__)\
)
#define BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG8B 1
#endif

#if defined(__x86_64__) && defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
#define BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG16B 1
#endif

#if defined(__x86_64__) || defined(__SSE2__)
#define BOOST_ATOMIC_DETAIL_X86_HAS_MFENCE 1
#endif

#else 

#if defined(__i386__) && !defined(BOOST_ATOMIC_NO_CMPXCHG8B)
#define BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG8B 1
#endif

#if defined(__x86_64__) && !defined(BOOST_ATOMIC_NO_CMPXCHG16B)
#define BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG16B 1
#endif

#if !defined(BOOST_ATOMIC_NO_MFENCE)
#define BOOST_ATOMIC_DETAIL_X86_HAS_MFENCE 1
#endif

#endif 

#define BOOST_ATOMIC_INT8_LOCK_FREE 2
#define BOOST_ATOMIC_INT16_LOCK_FREE 2
#define BOOST_ATOMIC_INT32_LOCK_FREE 2
#if defined(__x86_64__) || defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG8B)
#define BOOST_ATOMIC_INT64_LOCK_FREE 2
#endif
#if defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG16B)
#define BOOST_ATOMIC_INT128_LOCK_FREE 2
#endif
#define BOOST_ATOMIC_POINTER_LOCK_FREE 2

#define BOOST_ATOMIC_THREAD_FENCE 2
#define BOOST_ATOMIC_SIGNAL_FENCE 2

#endif 
