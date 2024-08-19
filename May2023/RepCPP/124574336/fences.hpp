


#ifndef BOOST_ATOMIC_FENCES_HPP_INCLUDED_
#define BOOST_ATOMIC_FENCES_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/capabilities.hpp>
#include <boost/atomic/detail/fence_operations.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif



namespace boost {

namespace atomics {

BOOST_FORCEINLINE void atomic_thread_fence(memory_order order) BOOST_NOEXCEPT
{
atomics::detail::fence_operations::thread_fence(order);
}

BOOST_FORCEINLINE void atomic_signal_fence(memory_order order) BOOST_NOEXCEPT
{
atomics::detail::fence_operations::signal_fence(order);
}

} 

using atomics::atomic_thread_fence;
using atomics::atomic_signal_fence;

} 

#include <boost/atomic/detail/footer.hpp>

#endif 
