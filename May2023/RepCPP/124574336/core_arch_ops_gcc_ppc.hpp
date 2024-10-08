


#ifndef BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_PPC_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_PPC_HPP_INCLUDED_

#include <cstddef>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/storage_traits.hpp>
#include <boost/atomic/detail/core_arch_operations_fwd.hpp>
#include <boost/atomic/detail/ops_gcc_ppc_common.hpp>
#include <boost/atomic/detail/capabilities.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {




template< bool Signed, bool Interprocess >
struct core_arch_operations< 4u, Signed, Interprocess > :
public core_arch_operations_gcc_ppc_base
{
typedef typename storage_traits< 4u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 4u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 4u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
__asm__ __volatile__
(
"stw %1, %0\n\t"
: "+m" (storage)
: "r" (v)
);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v;
if (order == memory_order_seq_cst)
__asm__ __volatile__ ("sync" ::: "memory");
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
{
__asm__ __volatile__
(
"lwz %0, %1\n\t"
"cmpw %0, %0\n\t"
"bne- 1f\n\t"
"1:\n\t"
"isync\n\t"
: "=&r" (v)
: "m" (storage)
: "cr0", "memory"
);
}
else
{
__asm__ __volatile__
(
"lwz %0, %1\n\t"
: "=&r" (v)
: "m" (storage)
);
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y1\n\t"
"stwcx. %2,%y1\n\t"
"bne- 1b\n\t"
: "=&b" (original), "+Z" (storage)
: "b" (v)
: "cr0"
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"lwarx %0,%y2\n\t"
"cmpw %0, %3\n\t"
"bne- 1f\n\t"
"stwcx. %4,%y2\n\t"
"bne- 1f\n\t"
"li %1, 1\n\t"
"1:\n\t"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"0: lwarx %0,%y2\n\t"
"cmpw %0, %3\n\t"
"bne- 1f\n\t"
"stwcx. %4,%y2\n\t"
"bne- 0b\n\t"
"li %1, 1\n\t"
"1:\n\t"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"and %1,%0,%3\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"or %1,%0,%3\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"xor %1,%0,%3\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool test_and_set(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
return !!exchange(storage, (storage_type)1, order);
}

static BOOST_FORCEINLINE void clear(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
store(storage, (storage_type)0, order);
}
};

#if defined(BOOST_ATOMIC_DETAIL_PPC_HAS_LBARX_STBCX)

template< bool Signed, bool Interprocess >
struct core_arch_operations< 1u, Signed, Interprocess > :
public core_arch_operations_gcc_ppc_base
{
typedef typename storage_traits< 1u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 1u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 1u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
__asm__ __volatile__
(
"stb %1, %0\n\t"
: "+m" (storage)
: "r" (v)
);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v;
if (order == memory_order_seq_cst)
__asm__ __volatile__ ("sync" ::: "memory");
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
{
__asm__ __volatile__
(
"lbz %0, %1\n\t"
"cmpw %0, %0\n\t"
"bne- 1f\n\t"
"1:\n\t"
"isync\n\t"
: "=&r" (v)
: "m" (storage)
: "cr0", "memory"
);
}
else
{
__asm__ __volatile__
(
"lbz %0, %1\n\t"
: "=&r" (v)
: "m" (storage)
);
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lbarx %0,%y1\n\t"
"stbcx. %2,%y1\n\t"
"bne- 1b\n\t"
: "=&b" (original), "+Z" (storage)
: "b" (v)
: "cr0"
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"lbarx %0,%y2\n\t"
"cmpw %0, %3\n\t"
"bne- 1f\n\t"
"stbcx. %4,%y2\n\t"
"bne- 1f\n\t"
"li %1, 1\n\t"
"1:\n\t"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"0: lbarx %0,%y2\n\t"
"cmpw %0, %3\n\t"
"bne- 1f\n\t"
"stbcx. %4,%y2\n\t"
"bne- 0b\n\t"
"li %1, 1\n\t"
"1:\n\t"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lbarx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"stbcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lbarx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"stbcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lbarx %0,%y2\n\t"
"and %1,%0,%3\n\t"
"stbcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lbarx %0,%y2\n\t"
"or %1,%0,%3\n\t"
"stbcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lbarx %0,%y2\n\t"
"xor %1,%0,%3\n\t"
"stbcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool test_and_set(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
return !!exchange(storage, (storage_type)1, order);
}

static BOOST_FORCEINLINE void clear(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
store(storage, (storage_type)0, order);
}
};

#else 

template< bool Interprocess >
struct core_arch_operations< 1u, false, Interprocess > :
public core_arch_operations< 4u, false, Interprocess >
{
typedef core_arch_operations< 4u, false, Interprocess > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"rlwinm %1, %1, 0, 0xff\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"rlwinm %1, %1, 0, 0xff\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};

template< bool Interprocess >
struct core_arch_operations< 1u, true, Interprocess > :
public core_arch_operations< 4u, true, Interprocess >
{
typedef core_arch_operations< 4u, true, Interprocess > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"extsb %1, %1\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"extsb %1, %1\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};

#endif 

#if defined(BOOST_ATOMIC_DETAIL_PPC_HAS_LHARX_STHCX)

template< bool Signed, bool Interprocess >
struct core_arch_operations< 2u, Signed, Interprocess > :
public core_arch_operations_gcc_ppc_base
{
typedef typename storage_traits< 2u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 2u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 2u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
__asm__ __volatile__
(
"sth %1, %0\n\t"
: "+m" (storage)
: "r" (v)
);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v;
if (order == memory_order_seq_cst)
__asm__ __volatile__ ("sync" ::: "memory");
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
{
__asm__ __volatile__
(
"lhz %0, %1\n\t"
"cmpw %0, %0\n\t"
"bne- 1f\n\t"
"1:\n\t"
"isync\n\t"
: "=&r" (v)
: "m" (storage)
: "cr0", "memory"
);
}
else
{
__asm__ __volatile__
(
"lhz %0, %1\n\t"
: "=&r" (v)
: "m" (storage)
);
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lharx %0,%y1\n\t"
"sthcx. %2,%y1\n\t"
"bne- 1b\n\t"
: "=&b" (original), "+Z" (storage)
: "b" (v)
: "cr0"
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"lharx %0,%y2\n\t"
"cmpw %0, %3\n\t"
"bne- 1f\n\t"
"sthcx. %4,%y2\n\t"
"bne- 1f\n\t"
"li %1, 1\n\t"
"1:\n\t"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"0: lharx %0,%y2\n\t"
"cmpw %0, %3\n\t"
"bne- 1f\n\t"
"sthcx. %4,%y2\n\t"
"bne- 0b\n\t"
"li %1, 1\n\t"
"1:\n\t"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lharx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"sthcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lharx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"sthcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lharx %0,%y2\n\t"
"and %1,%0,%3\n\t"
"sthcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lharx %0,%y2\n\t"
"or %1,%0,%3\n\t"
"sthcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lharx %0,%y2\n\t"
"xor %1,%0,%3\n\t"
"sthcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool test_and_set(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
return !!exchange(storage, (storage_type)1, order);
}

static BOOST_FORCEINLINE void clear(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
store(storage, (storage_type)0, order);
}
};

#else 

template< bool Interprocess >
struct core_arch_operations< 2u, false, Interprocess > :
public core_arch_operations< 4u, false, Interprocess >
{
typedef core_arch_operations< 4u, false, Interprocess > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"rlwinm %1, %1, 0, 0xffff\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"rlwinm %1, %1, 0, 0xffff\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};

template< bool Interprocess >
struct core_arch_operations< 2u, true, Interprocess > :
public core_arch_operations< 4u, true, Interprocess >
{
typedef core_arch_operations< 4u, true, Interprocess > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"extsh %1, %1\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"lwarx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"extsh %1, %1\n\t"
"stwcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};

#endif 

#if defined(BOOST_ATOMIC_DETAIL_PPC_HAS_LDARX_STDCX)

template< bool Signed, bool Interprocess >
struct core_arch_operations< 8u, Signed, Interprocess > :
public core_arch_operations_gcc_ppc_base
{
typedef typename storage_traits< 8u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 8u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 8u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
__asm__ __volatile__
(
"std %1, %0\n\t"
: "+m" (storage)
: "r" (v)
);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v;
if (order == memory_order_seq_cst)
__asm__ __volatile__ ("sync" ::: "memory");
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
{
__asm__ __volatile__
(
"ld %0, %1\n\t"
"cmpd %0, %0\n\t"
"bne- 1f\n\t"
"1:\n\t"
"isync\n\t"
: "=&b" (v)
: "m" (storage)
: "cr0", "memory"
);
}
else
{
__asm__ __volatile__
(
"ld %0, %1\n\t"
: "=&b" (v)
: "m" (storage)
);
}
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldarx %0,%y1\n\t"
"stdcx. %2,%y1\n\t"
"bne- 1b\n\t"
: "=&b" (original), "+Z" (storage)
: "b" (v)
: "cr0"
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"ldarx %0,%y2\n\t"
"cmpd %0, %3\n\t"
"bne- 1f\n\t"
"stdcx. %4,%y2\n\t"
"bne- 1f\n\t"
"li %1, 1\n\t"
"1:"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
int success;
fence_before(success_order);
__asm__ __volatile__
(
"li %1, 0\n\t"
"0: ldarx %0,%y2\n\t"
"cmpd %0, %3\n\t"
"bne- 1f\n\t"
"stdcx. %4,%y2\n\t"
"bne- 0b\n\t"
"li %1, 1\n\t"
"1:\n\t"
: "=&b" (expected), "=&b" (success), "+Z" (storage)
: "b" (expected), "b" (desired)
: "cr0"
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldarx %0,%y2\n\t"
"add %1,%0,%3\n\t"
"stdcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldarx %0,%y2\n\t"
"sub %1,%0,%3\n\t"
"stdcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldarx %0,%y2\n\t"
"and %1,%0,%3\n\t"
"stdcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldarx %0,%y2\n\t"
"or %1,%0,%3\n\t"
"stdcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, result;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldarx %0,%y2\n\t"
"xor %1,%0,%3\n\t"
"stdcx. %1,%y2\n\t"
"bne- 1b\n\t"
: "=&b" (original), "=&b" (result), "+Z" (storage)
: "b" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool test_and_set(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
return !!exchange(storage, (storage_type)1, order);
}

static BOOST_FORCEINLINE void clear(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
store(storage, (storage_type)0, order);
}
};

#endif 

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
