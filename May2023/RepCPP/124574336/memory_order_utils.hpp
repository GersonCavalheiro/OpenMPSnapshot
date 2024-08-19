


#ifndef BOOST_ATOMIC_DETAIL_MEMORY_ORDER_UTILS_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_MEMORY_ORDER_UTILS_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

BOOST_FORCEINLINE BOOST_CONSTEXPR memory_order deduce_failure_order(memory_order order) BOOST_NOEXCEPT
{
return order == memory_order_acq_rel ? memory_order_acquire : (order == memory_order_release ? memory_order_relaxed : order);
}

BOOST_FORCEINLINE BOOST_CONSTEXPR bool cas_failure_order_must_not_be_stronger_than_success_order(memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
return (static_cast< unsigned int >(failure_order) & 15u) <= (static_cast< unsigned int >(success_order) & 15u);
}

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
