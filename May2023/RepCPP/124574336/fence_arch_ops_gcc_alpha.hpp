


#ifndef BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_GCC_ALPHA_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_GCC_ALPHA_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

struct fence_arch_operations_gcc_alpha
{
static BOOST_FORCEINLINE void thread_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
__asm__ __volatile__ ("mb" ::: "memory");
}

static BOOST_FORCEINLINE void signal_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
__asm__ __volatile__ ("" ::: "memory");
}
};

typedef fence_arch_operations_gcc_alpha fence_arch_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
