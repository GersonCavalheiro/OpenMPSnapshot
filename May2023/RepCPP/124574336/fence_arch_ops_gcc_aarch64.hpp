


#ifndef BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_GCC_AARCH64_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_GCC_AARCH64_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

struct fence_arch_operations_gcc_aarch64
{
static BOOST_FORCEINLINE void thread_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
{
if (order == memory_order_consume || order == memory_order_acquire)
__asm__ __volatile__ ("dmb ishld\n\t" ::: "memory");
else
__asm__ __volatile__ ("dmb ish\n\t" ::: "memory");
}
}

static BOOST_FORCEINLINE void signal_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
__asm__ __volatile__ ("" ::: "memory");
}
};

typedef fence_arch_operations_gcc_aarch64 fence_arch_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
