


#ifndef BOOST_ATOMIC_DETAIL_FUTEX_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FUTEX_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(__linux__) || defined(__OpenBSD__) || defined(__NETBSD__) || defined(__NetBSD__)

#include <sys/syscall.h>

#if defined(SYS_futex)
#define BOOST_ATOMIC_DETAIL_SYS_FUTEX SYS_futex
#elif defined(__NR_futex)
#define BOOST_ATOMIC_DETAIL_SYS_FUTEX __NR_futex
#elif defined(SYS___futex)
#define BOOST_ATOMIC_DETAIL_SYS_FUTEX SYS___futex
#define BOOST_ATOMIC_DETAIL_NETBSD_FUTEX
#endif

#if defined(BOOST_ATOMIC_DETAIL_SYS_FUTEX)

#include <cstddef>
#if defined(__linux__)
#include <linux/futex.h>
#else
#include <sys/futex.h>
#endif
#include <boost/atomic/detail/intptr.hpp>
#include <boost/atomic/detail/header.hpp>

#define BOOST_ATOMIC_DETAIL_HAS_FUTEX

#if defined(FUTEX_PRIVATE_FLAG)
#define BOOST_ATOMIC_DETAIL_FUTEX_PRIVATE_FLAG FUTEX_PRIVATE_FLAG
#else
#define BOOST_ATOMIC_DETAIL_FUTEX_PRIVATE_FLAG 0
#endif

namespace boost {
namespace atomics {
namespace detail {

BOOST_FORCEINLINE int futex_invoke(void* addr1, int op, unsigned int val1, const void* timeout = NULL, void* addr2 = NULL, unsigned int val3 = 0) BOOST_NOEXCEPT
{
#if !defined(BOOST_ATOMIC_DETAIL_NETBSD_FUTEX)
return ::syscall(BOOST_ATOMIC_DETAIL_SYS_FUTEX, addr1, op, val1, timeout, addr2, val3);
#else
return ::syscall(BOOST_ATOMIC_DETAIL_SYS_FUTEX, addr1, op, val1, timeout, addr2, 0u, val3);
#endif
}

BOOST_FORCEINLINE int futex_invoke(void* addr1, int op, unsigned int val1, unsigned int val2, void* addr2 = NULL, unsigned int val3 = 0) BOOST_NOEXCEPT
{
#if !defined(BOOST_ATOMIC_DETAIL_NETBSD_FUTEX)
return ::syscall(BOOST_ATOMIC_DETAIL_SYS_FUTEX, addr1, op, val1, static_cast< atomics::detail::uintptr_t >(val2), addr2, val3);
#else
return ::syscall(BOOST_ATOMIC_DETAIL_SYS_FUTEX, addr1, op, val1, static_cast< void* >(NULL), addr2, val2, val3);
#endif
}

BOOST_FORCEINLINE int futex_wait(void* pval, unsigned int expected) BOOST_NOEXCEPT
{
return futex_invoke(pval, FUTEX_WAIT, expected);
}

BOOST_FORCEINLINE int futex_wait_private(void* pval, unsigned int expected) BOOST_NOEXCEPT
{
return futex_invoke(pval, FUTEX_WAIT | BOOST_ATOMIC_DETAIL_FUTEX_PRIVATE_FLAG, expected);
}

BOOST_FORCEINLINE int futex_signal(void* pval, unsigned int count = 1u) BOOST_NOEXCEPT
{
return futex_invoke(pval, FUTEX_WAKE, count);
}

BOOST_FORCEINLINE int futex_signal_private(void* pval, unsigned int count = 1u) BOOST_NOEXCEPT
{
return futex_invoke(pval, FUTEX_WAKE | BOOST_ATOMIC_DETAIL_FUTEX_PRIVATE_FLAG, count);
}

BOOST_FORCEINLINE int futex_broadcast(void* pval) BOOST_NOEXCEPT
{
return futex_signal(pval, (~static_cast< unsigned int >(0u)) >> 1);
}

BOOST_FORCEINLINE int futex_broadcast_private(void* pval) BOOST_NOEXCEPT
{
return futex_signal_private(pval, (~static_cast< unsigned int >(0u)) >> 1);
}

BOOST_FORCEINLINE int futex_requeue(void* pval1, void* pval2, unsigned int wake_count = 1u, unsigned int requeue_count = (~static_cast< unsigned int >(0u)) >> 1) BOOST_NOEXCEPT
{
return futex_invoke(pval1, FUTEX_REQUEUE, wake_count, requeue_count, pval2);
}

BOOST_FORCEINLINE int futex_requeue_private(void* pval1, void* pval2, unsigned int wake_count = 1u, unsigned int requeue_count = (~static_cast< unsigned int >(0u)) >> 1) BOOST_NOEXCEPT
{
return futex_invoke(pval1, FUTEX_REQUEUE | BOOST_ATOMIC_DETAIL_FUTEX_PRIVATE_FLAG, wake_count, requeue_count, pval2);
}

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 

#endif 

#endif 
