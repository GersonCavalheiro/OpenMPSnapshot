


#ifndef BOOST_ATOMIC_DETAIL_CAS_BASED_EXCHANGE_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CAS_BASED_EXCHANGE_HPP_INCLUDED_

#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

template< typename Base >
struct cas_based_exchange :
public Base
{
typedef typename Base::storage_type storage_type;

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type old_val;
atomics::detail::non_atomic_load(storage, old_val);
while (!Base::compare_exchange_weak(storage, old_val, v, order, memory_order_relaxed)) {}
return old_val;
}
};

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
