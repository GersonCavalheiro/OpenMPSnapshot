


#ifndef BOOST_ATOMIC_DETAIL_CORE_OPS_LINUX_ARM_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_OPS_LINUX_ARM_HPP_INCLUDED_

#include <cstddef>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/storage_traits.hpp>
#include <boost/atomic/detail/core_operations_fwd.hpp>
#include <boost/atomic/detail/core_ops_cas_based.hpp>
#include <boost/atomic/detail/cas_based_exchange.hpp>
#include <boost/atomic/detail/extending_cas_based_arithmetic.hpp>
#include <boost/atomic/detail/fence_operations.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {


struct linux_arm_cas_base
{
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = true;
static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void fence_before_store(memory_order order) BOOST_NOEXCEPT
{
if ((static_cast< unsigned int >(order) & static_cast< unsigned int >(memory_order_release)) != 0u)
fence_operations::hardware_full_fence();
}

static BOOST_FORCEINLINE void fence_after_store(memory_order order) BOOST_NOEXCEPT
{
if (order == memory_order_seq_cst)
fence_operations::hardware_full_fence();
}

static BOOST_FORCEINLINE void fence_after_load(memory_order order) BOOST_NOEXCEPT
{
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
fence_operations::hardware_full_fence();
}
};

template< bool Signed, bool Interprocess >
struct linux_arm_cas :
public linux_arm_cas_base
{
typedef typename storage_traits< 4u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 4u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 4u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before_store(order);
storage = v;
fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = storage;
fence_after_load(order);
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
while (true)
{
storage_type tmp = expected;
if (compare_exchange_weak(storage, tmp, desired, success_order, failure_order))
return true;
if (tmp != expected)
{
expected = tmp;
return false;
}
}
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order, memory_order) BOOST_NOEXCEPT
{
typedef storage_type (*kernel_cmpxchg32_t)(storage_type oldval, storage_type newval, volatile storage_type* ptr);

if (((kernel_cmpxchg32_t)0xffff0fc0)(expected, desired, &storage) == 0)
{
return true;
}
else
{
expected = storage;
return false;
}
}
};

template< bool Signed, bool Interprocess >
struct core_operations< 1u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_cas_based< cas_based_exchange< linux_arm_cas< Signed, Interprocess > > >, 1u, Signed >
{
};

template< bool Signed, bool Interprocess >
struct core_operations< 2u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_cas_based< cas_based_exchange< linux_arm_cas< Signed, Interprocess > > >, 2u, Signed >
{
};

template< bool Signed, bool Interprocess >
struct core_operations< 4u, Signed, Interprocess > :
public core_operations_cas_based< cas_based_exchange< linux_arm_cas< Signed, Interprocess > > >
{
};

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
