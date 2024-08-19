


#ifndef BOOST_ATOMIC_DETAIL_OPS_GCC_PPC_COMMON_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_OPS_GCC_PPC_COMMON_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {



struct core_arch_operations_gcc_ppc_base
{
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = false;
static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void fence_before(memory_order order) BOOST_NOEXCEPT
{
#if defined(__powerpc64__) || defined(__PPC64__)
if (order == memory_order_seq_cst)
__asm__ __volatile__ ("sync" ::: "memory");
else if ((static_cast< unsigned int >(order) & static_cast< unsigned int >(memory_order_release)) != 0u)
__asm__ __volatile__ ("lwsync" ::: "memory");
#else
if ((static_cast< unsigned int >(order) & static_cast< unsigned int >(memory_order_release)) != 0u)
__asm__ __volatile__ ("sync" ::: "memory");
#endif
}

static BOOST_FORCEINLINE void fence_after(memory_order order) BOOST_NOEXCEPT
{
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
__asm__ __volatile__ ("isync" ::: "memory");
}
};

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
