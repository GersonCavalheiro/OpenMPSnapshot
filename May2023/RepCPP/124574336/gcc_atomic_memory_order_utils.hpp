


#ifndef BOOST_ATOMIC_DETAIL_GCC_ATOMIC_MEMORY_ORDER_UTILS_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_GCC_ATOMIC_MEMORY_ORDER_UTILS_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {


BOOST_FORCEINLINE BOOST_CONSTEXPR int convert_memory_order_to_gcc(memory_order order) BOOST_NOEXCEPT
{
return (order == memory_order_relaxed ? __ATOMIC_RELAXED : (order == memory_order_consume ? __ATOMIC_CONSUME :
(order == memory_order_acquire ? __ATOMIC_ACQUIRE : (order == memory_order_release ? __ATOMIC_RELEASE :
(order == memory_order_acq_rel ? __ATOMIC_ACQ_REL : __ATOMIC_SEQ_CST)))));
}

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
