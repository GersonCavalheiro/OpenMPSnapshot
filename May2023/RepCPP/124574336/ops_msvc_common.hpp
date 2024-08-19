


#ifndef BOOST_ATOMIC_DETAIL_OPS_MSVC_COMMON_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_OPS_MSVC_COMMON_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(__INTEL_COMPILER)
#define BOOST_ATOMIC_DETAIL_COMPILER_BARRIER() __memory_barrier()
#elif defined(__clang__)
#define BOOST_ATOMIC_DETAIL_COMPILER_BARRIER() __atomic_signal_fence(__ATOMIC_SEQ_CST)
#elif defined(_MSC_VER) && !defined(_WIN32_WCE)
extern "C" void _ReadWriteBarrier(void);
#pragma intrinsic(_ReadWriteBarrier)
#define BOOST_ATOMIC_DETAIL_COMPILER_BARRIER() _ReadWriteBarrier()
#endif

#ifndef BOOST_ATOMIC_DETAIL_COMPILER_BARRIER
#define BOOST_ATOMIC_DETAIL_COMPILER_BARRIER()
#endif

#include <boost/atomic/detail/footer.hpp>

#endif 
