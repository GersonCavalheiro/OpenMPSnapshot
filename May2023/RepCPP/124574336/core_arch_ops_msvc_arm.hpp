


#ifndef BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_MSVC_ARM_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_MSVC_ARM_HPP_INCLUDED_

#include <cstddef>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/interlocked.hpp>
#include <boost/atomic/detail/storage_traits.hpp>
#include <boost/atomic/detail/core_arch_operations_fwd.hpp>
#include <boost/atomic/detail/type_traits/make_signed.hpp>
#include <boost/atomic/detail/ops_msvc_common.hpp>
#include <boost/atomic/detail/fence_arch_operations.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

extern "C" {
__int8 __iso_volatile_load8(const volatile __int8*);
__int16 __iso_volatile_load16(const volatile __int16*);
__int32 __iso_volatile_load32(const volatile __int32*);
__int64 __iso_volatile_load64(const volatile __int64*);
void __iso_volatile_store8(volatile __int8*, __int8);
void __iso_volatile_store16(volatile __int16*, __int16);
void __iso_volatile_store32(volatile __int32*, __int32);
void __iso_volatile_store64(volatile __int64*, __int64);
}
#if defined(BOOST_MSVC)
#pragma intrinsic(__iso_volatile_load8)
#pragma intrinsic(__iso_volatile_load16)
#pragma intrinsic(__iso_volatile_load32)
#pragma intrinsic(__iso_volatile_load64)
#pragma intrinsic(__iso_volatile_store8)
#pragma intrinsic(__iso_volatile_store16)
#pragma intrinsic(__iso_volatile_store32)
#pragma intrinsic(__iso_volatile_store64)
#endif

#define BOOST_ATOMIC_DETAIL_ARM_LOAD8(p) __iso_volatile_load8((const volatile __int8*)(p))
#define BOOST_ATOMIC_DETAIL_ARM_LOAD16(p) __iso_volatile_load16((const volatile __int16*)(p))
#define BOOST_ATOMIC_DETAIL_ARM_LOAD32(p) __iso_volatile_load32((const volatile __int32*)(p))
#define BOOST_ATOMIC_DETAIL_ARM_LOAD64(p) __iso_volatile_load64((const volatile __int64*)(p))
#define BOOST_ATOMIC_DETAIL_ARM_STORE8(p, v) __iso_volatile_store8((volatile __int8*)(p), (__int8)(v))
#define BOOST_ATOMIC_DETAIL_ARM_STORE16(p, v) __iso_volatile_store16((volatile __int16*)(p), (__int16)(v))
#define BOOST_ATOMIC_DETAIL_ARM_STORE32(p, v) __iso_volatile_store32((volatile __int32*)(p), (__int32)(v))
#define BOOST_ATOMIC_DETAIL_ARM_STORE64(p, v) __iso_volatile_store64((volatile __int64*)(p), (__int64)(v))

namespace boost {
namespace atomics {
namespace detail {


struct core_arch_operations_msvc_arm_base
{
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = false;
static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void fence_before_store(memory_order order) BOOST_NOEXCEPT
{
BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();

if ((static_cast< unsigned int >(order) & static_cast< unsigned int >(memory_order_release)) != 0u)
fence_arch_operations::hardware_full_fence();

BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();
}

static BOOST_FORCEINLINE void fence_after_store(memory_order order) BOOST_NOEXCEPT
{
BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();

if (order == memory_order_seq_cst)
fence_arch_operations::hardware_full_fence();

BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();
}

static BOOST_FORCEINLINE void fence_after_load(memory_order order) BOOST_NOEXCEPT
{
BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();

if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
fence_arch_operations::hardware_full_fence();

BOOST_ATOMIC_DETAIL_COMPILER_BARRIER();
}

static BOOST_FORCEINLINE BOOST_CONSTEXPR memory_order cas_common_order(memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
return static_cast< memory_order >(((static_cast< unsigned int >(failure_order) | static_cast< unsigned int >(success_order)) & ~static_cast< unsigned int >(memory_order_consume))
| (((static_cast< unsigned int >(failure_order) | static_cast< unsigned int >(success_order)) & static_cast< unsigned int >(memory_order_consume)) << 1u));
}
};

template< std::size_t Size, bool Signed, bool Interprocess, typename Derived >
struct core_arch_operations_msvc_arm :
public core_arch_operations_msvc_arm_base
{
typedef typename storage_traits< Size >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = Size;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = storage_traits< Size >::alignment;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
typedef typename boost::atomics::detail::make_signed< storage_type >::type signed_storage_type;
return Derived::fetch_add(storage, static_cast< storage_type >(-static_cast< signed_storage_type >(v)), order);
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
return Derived::compare_exchange_strong(storage, expected, desired, success_order, failure_order);
}

static BOOST_FORCEINLINE bool test_and_set(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
return !!Derived::exchange(storage, (storage_type)1, order);
}

static BOOST_FORCEINLINE void clear(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
Derived::store(storage, (storage_type)0, order);
}
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 1u, Signed, Interprocess > :
public core_arch_operations_msvc_arm< 1u, Signed, Interprocess, core_arch_operations< 1u, Signed, Interprocess > >
{
typedef core_arch_operations_msvc_arm< 1u, Signed, Interprocess, core_arch_operations< 1u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before_store(order);
BOOST_ATOMIC_DETAIL_ARM_STORE8(&storage, v);
base_type::fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = BOOST_ATOMIC_DETAIL_ARM_LOAD8(&storage);
base_type::fence_after_load(order);
return v;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD8_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD8_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD8_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD8(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE8_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE8_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE8_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE8(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
storage_type previous = expected, old_val;

switch (cas_common_order(success_order, failure_order))
{
case memory_order_relaxed:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE8_RELAXED(&storage, desired, previous));
break;
case memory_order_consume:
case memory_order_acquire:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE8_ACQUIRE(&storage, desired, previous));
break;
case memory_order_release:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE8_RELEASE(&storage, desired, previous));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE8(&storage, desired, previous));
break;
}
expected = old_val;

return (previous == old_val);
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND8_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND8_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND8_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND8(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR8_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR8_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR8_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR8(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR8_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR8_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR8_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR8(&storage, v));
break;
}
return v;
}
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 2u, Signed, Interprocess > :
public core_arch_operations_msvc_arm< 2u, Signed, Interprocess, core_arch_operations< 2u, Signed, Interprocess > >
{
typedef core_arch_operations_msvc_arm< 2u, Signed, Interprocess, core_arch_operations< 2u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before_store(order);
BOOST_ATOMIC_DETAIL_ARM_STORE16(&storage, v);
base_type::fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = BOOST_ATOMIC_DETAIL_ARM_LOAD16(&storage);
base_type::fence_after_load(order);
return v;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD16_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD16_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD16_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD16(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE16_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE16_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE16_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE16(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
storage_type previous = expected, old_val;

switch (cas_common_order(success_order, failure_order))
{
case memory_order_relaxed:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE16_RELAXED(&storage, desired, previous));
break;
case memory_order_consume:
case memory_order_acquire:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE16_ACQUIRE(&storage, desired, previous));
break;
case memory_order_release:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE16_RELEASE(&storage, desired, previous));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE16(&storage, desired, previous));
break;
}
expected = old_val;

return (previous == old_val);
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND16_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND16_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND16_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND16(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR16_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR16_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR16_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR16(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR16_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR16_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR16_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR16(&storage, v));
break;
}
return v;
}
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 4u, Signed, Interprocess > :
public core_arch_operations_msvc_arm< 4u, Signed, Interprocess, core_arch_operations< 4u, Signed, Interprocess > >
{
typedef core_arch_operations_msvc_arm< 4u, Signed, Interprocess, core_arch_operations< 4u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before_store(order);
BOOST_ATOMIC_DETAIL_ARM_STORE32(&storage, v);
base_type::fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = BOOST_ATOMIC_DETAIL_ARM_LOAD32(&storage);
base_type::fence_after_load(order);
return v;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
storage_type previous = expected, old_val;

switch (cas_common_order(success_order, failure_order))
{
case memory_order_relaxed:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE_RELAXED(&storage, desired, previous));
break;
case memory_order_consume:
case memory_order_acquire:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE_ACQUIRE(&storage, desired, previous));
break;
case memory_order_release:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE_RELEASE(&storage, desired, previous));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE(&storage, desired, previous));
break;
}
expected = old_val;

return (previous == old_val);
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR(&storage, v));
break;
}
return v;
}
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 8u, Signed, Interprocess > :
public core_arch_operations_msvc_arm< 8u, Signed, Interprocess, core_arch_operations< 8u, Signed, Interprocess > >
{
typedef core_arch_operations_msvc_arm< 8u, Signed, Interprocess, core_arch_operations< 8u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before_store(order);
BOOST_ATOMIC_DETAIL_ARM_STORE64(&storage, v);
base_type::fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = BOOST_ATOMIC_DETAIL_ARM_LOAD64(&storage);
base_type::fence_after_load(order);
return v;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD64_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD64_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD64_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE_ADD64(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE64_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE64_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE64_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_EXCHANGE64(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
storage_type previous = expected, old_val;

switch (cas_common_order(success_order, failure_order))
{
case memory_order_relaxed:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE64_RELAXED(&storage, desired, previous));
break;
case memory_order_consume:
case memory_order_acquire:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE64_ACQUIRE(&storage, desired, previous));
break;
case memory_order_release:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE64_RELEASE(&storage, desired, previous));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
old_val = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_COMPARE_EXCHANGE64(&storage, desired, previous));
break;
}
expected = old_val;

return (previous == old_val);
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND64_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND64_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND64_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_AND64(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR64_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR64_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR64_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_OR64(&storage, v));
break;
}
return v;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
switch (order)
{
case memory_order_relaxed:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR64_RELAXED(&storage, v));
break;
case memory_order_consume:
case memory_order_acquire:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR64_ACQUIRE(&storage, v));
break;
case memory_order_release:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR64_RELEASE(&storage, v));
break;
case memory_order_acq_rel:
case memory_order_seq_cst:
default:
v = static_cast< storage_type >(BOOST_ATOMIC_INTERLOCKED_XOR64(&storage, v));
break;
}
return v;
}
};

} 
} 
} 

#undef BOOST_ATOMIC_DETAIL_ARM_LOAD8
#undef BOOST_ATOMIC_DETAIL_ARM_LOAD16
#undef BOOST_ATOMIC_DETAIL_ARM_LOAD32
#undef BOOST_ATOMIC_DETAIL_ARM_LOAD64
#undef BOOST_ATOMIC_DETAIL_ARM_STORE8
#undef BOOST_ATOMIC_DETAIL_ARM_STORE16
#undef BOOST_ATOMIC_DETAIL_ARM_STORE32
#undef BOOST_ATOMIC_DETAIL_ARM_STORE64

#include <boost/atomic/detail/footer.hpp>

#endif 
