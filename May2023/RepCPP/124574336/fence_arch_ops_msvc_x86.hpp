


#ifndef BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_MSVC_X86_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPS_MSVC_X86_HPP_INCLUDED_

#include <boost/cstdint.hpp>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/interlocked.hpp>
#include <boost/atomic/detail/ops_msvc_common.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

struct fence_arch_operations_msvc_x86
{
static BOOST_FORCEINLINE void thread_fence(memory_order order) BOOST_NOEXCEPT
{
if (order == memory_order_seq_cst)
{
boost::uint32_t dummy;
BOOST_ATOMIC_INTERLOCKED_INCREMENT(&dummy);
}
else if (order != memory_order_relaxed)
{
BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();
}
}

static BOOST_FORCEINLINE void signal_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();
}
};

typedef fence_arch_operations_msvc_x86 fence_arch_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
