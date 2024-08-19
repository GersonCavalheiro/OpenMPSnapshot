


#ifndef BOOST_ATOMIC_DETAIL_CORE_OPS_GCC_ATOMIC_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_OPS_GCC_ATOMIC_HPP_INCLUDED_

#include <cstddef>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/storage_traits.hpp>
#include <boost/atomic/detail/core_operations_fwd.hpp>
#include <boost/atomic/detail/core_arch_operations.hpp>
#include <boost/atomic/detail/capabilities.hpp>
#include <boost/atomic/detail/gcc_atomic_memory_order_utils.hpp>

#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT8_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE || BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE ||\
BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE || BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE < BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE
#include <boost/atomic/detail/int_sizes.hpp>
#include <boost/atomic/detail/extending_cas_based_arithmetic.hpp>
#endif
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

template< std::size_t Size, bool Signed, bool Interprocess >
struct core_operations_gcc_atomic
{
typedef typename storage_traits< Size >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = Size;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = storage_traits< Size >::alignment;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = false;

static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
#if defined(BOOST_GCC) && BOOST_GCC < 100100 && (defined(__x86_64__) || defined(__i386__))
if (order != memory_order_seq_cst)
__atomic_store_n(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
else
__atomic_exchange_n(&storage, v, __ATOMIC_SEQ_CST);
#else
__atomic_store_n(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
#endif
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
#if defined(BOOST_ATOMIC_DETAIL_AARCH64_HAS_RCPC)
typedef atomics::detail::core_arch_operations< storage_size, is_signed, is_interprocess > core_arch_operations;
return core_arch_operations::load(storage, order);
#else
return __atomic_load_n(&storage, atomics::detail::convert_memory_order_to_gcc(order));
#endif
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
return __atomic_fetch_add(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
return __atomic_fetch_sub(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
return __atomic_exchange_n(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
return __atomic_compare_exchange_n
(
&storage, &expected, desired, false,
atomics::detail::convert_memory_order_to_gcc(success_order),
atomics::detail::convert_memory_order_to_gcc(failure_order)
);
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
return __atomic_compare_exchange_n
(
&storage, &expected, desired, true,
atomics::detail::convert_memory_order_to_gcc(success_order),
atomics::detail::convert_memory_order_to_gcc(failure_order)
);
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
return __atomic_fetch_and(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
return __atomic_fetch_or(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
return __atomic_fetch_xor(&storage, v, atomics::detail::convert_memory_order_to_gcc(order));
}

static BOOST_FORCEINLINE bool test_and_set(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
return __atomic_test_and_set(&storage, atomics::detail::convert_memory_order_to_gcc(order));
}

static BOOST_FORCEINLINE void clear(storage_type volatile& storage, memory_order order) BOOST_NOEXCEPT
{
__atomic_clear(const_cast< storage_type* >(&storage), atomics::detail::convert_memory_order_to_gcc(order));
}
};


#if BOOST_ATOMIC_INT128_LOCK_FREE > 0 && BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE >= BOOST_ATOMIC_INT128_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 16u, Signed, Interprocess > :
public core_operations_gcc_atomic< 16u, Signed, Interprocess >
{
};

#endif

#if BOOST_ATOMIC_INT64_LOCK_FREE > 0
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE >= BOOST_ATOMIC_INT64_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 8u, Signed, Interprocess > :
public core_operations_gcc_atomic< 8u, Signed, Interprocess >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE >= BOOST_ATOMIC_INT64_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 8u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 16u, Signed, Interprocess >, 8u, Signed >
{
};

#endif
#endif 


#if BOOST_ATOMIC_INT32_LOCK_FREE > 0
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE >= BOOST_ATOMIC_INT32_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 4u, Signed, Interprocess > :
public core_operations_gcc_atomic< 4u, Signed, Interprocess >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE >= BOOST_ATOMIC_INT32_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 4u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 8u, Signed, Interprocess >, 4u, Signed >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE >= BOOST_ATOMIC_INT32_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 8u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 16u, Signed, Interprocess >, 4u, Signed >
{
};

#endif
#endif 


#if BOOST_ATOMIC_INT16_LOCK_FREE > 0
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE >= BOOST_ATOMIC_INT16_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 2u, Signed, Interprocess > :
public core_operations_gcc_atomic< 2u, Signed, Interprocess >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE >= BOOST_ATOMIC_INT16_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 2u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 4u, Signed, Interprocess >, 2u, Signed >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE >= BOOST_ATOMIC_INT16_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 2u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 8u, Signed, Interprocess >, 2u, Signed >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE >= BOOST_ATOMIC_INT16_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 2u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 16u, Signed, Interprocess >, 2u, Signed >
{
};

#endif
#endif 


#if BOOST_ATOMIC_INT8_LOCK_FREE > 0
#if BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT8_LOCK_FREE >= BOOST_ATOMIC_INT8_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 1u, Signed, Interprocess > :
public core_operations_gcc_atomic< 1u, Signed, Interprocess >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT16_LOCK_FREE >= BOOST_ATOMIC_INT8_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 1u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 2u, Signed, Interprocess >, 1u, Signed >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT32_LOCK_FREE >= BOOST_ATOMIC_INT8_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 1u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 4u, Signed, Interprocess >, 1u, Signed >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT64_LOCK_FREE >= BOOST_ATOMIC_INT8_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 1u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 8u, Signed, Interprocess >, 1u, Signed >
{
};

#elif BOOST_ATOMIC_DETAIL_GCC_ATOMIC_INT128_LOCK_FREE >= BOOST_ATOMIC_INT8_LOCK_FREE

template< bool Signed, bool Interprocess >
struct core_operations< 1u, Signed, Interprocess > :
public extending_cas_based_arithmetic< core_operations_gcc_atomic< 16u, Signed, Interprocess >, 1u, Signed >
{
};

#endif
#endif 

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
