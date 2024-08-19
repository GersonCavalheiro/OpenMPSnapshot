


#ifndef BOOST_ATOMIC_DETAIL_FENCE_OPS_LINUX_ARM_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_OPS_LINUX_ARM_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

struct fence_operations_linux_arm
{
static BOOST_FORCEINLINE void thread_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
hardware_full_fence();
}

static BOOST_FORCEINLINE void signal_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
__asm__ __volatile__ ("" ::: "memory");
}

static BOOST_FORCEINLINE void hardware_full_fence() BOOST_NOEXCEPT
{
typedef void (*kernel_dmb_t)(void);
((kernel_dmb_t)0xffff0fa0)();
}
};

typedef fence_operations_linux_arm fence_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
