


#ifndef BOOST_ATOMIC_DETAIL_FENCE_OPS_GCC_SYNC_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_OPS_GCC_SYNC_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

struct fence_operations_gcc_sync
{
static BOOST_FORCEINLINE void thread_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
__sync_synchronize();
}

static BOOST_FORCEINLINE void signal_fence(memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_relaxed)
__asm__ __volatile__ ("" ::: "memory");
}
};

typedef fence_operations_gcc_sync fence_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
