


#ifndef BOOST_ATOMIC_DETAIL_FENCE_OPS_GCC_ATOMIC_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_OPS_GCC_ATOMIC_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/fence_arch_operations.hpp>
#include <boost/atomic/detail/gcc_atomic_memory_order_utils.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(__INTEL_COMPILER)
#pragma system_header
#endif

namespace boost {
namespace atomics {
namespace detail {

struct fence_operations_gcc_atomic
{
static BOOST_FORCEINLINE void thread_fence(memory_order order) BOOST_NOEXCEPT
{
#if defined(__x86_64__) || defined(__i386__)
if (order != memory_order_seq_cst)
{
__atomic_thread_fence(atomics::detail::convert_memory_order_to_gcc(order));
}
else
{
fence_arch_operations::thread_fence(order);
}
#else
__atomic_thread_fence(atomics::detail::convert_memory_order_to_gcc(order));
#endif
}

static BOOST_FORCEINLINE void signal_fence(memory_order order) BOOST_NOEXCEPT
{
__atomic_signal_fence(atomics::detail::convert_memory_order_to_gcc(order));
}
};

typedef fence_operations_gcc_atomic fence_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
