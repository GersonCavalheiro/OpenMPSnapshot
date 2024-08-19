


#ifndef BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_GCC_ARM_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_GCC_ARM_HPP_INCLUDED_

#include <boost/cstdint.hpp>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/capabilities.hpp>
#include <boost/atomic/detail/gcc_arm_asm_common.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

struct fence_arch_operations_gcc_arm
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

#if defined(BOOST_ATOMIC_DETAIL_ARM_HAS_DMB)
__asm__ __volatile__
(
#if defined(__thumb2__)
".short 0xF3BF, 0x8F5B\n\t" 
#else
".word 0xF57FF05B\n\t" 
#endif
:
:
: "memory"
);
#else
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"mcr p15, 0, r0, c7, c10, 5\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: "=&l" (tmp)
:
: "memory"
);
#endif
}
};

typedef fence_arch_operations_gcc_arm fence_arch_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
