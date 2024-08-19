


#ifndef BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_X86_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_X86_HPP_INCLUDED_

#include <cstddef>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/storage_traits.hpp>
#include <boost/atomic/detail/core_arch_operations_fwd.hpp>
#include <boost/atomic/detail/capabilities.hpp>
#if defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG8B) || defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG16B)
#include <boost/cstdint.hpp>
#include <boost/atomic/detail/intptr.hpp>
#include <boost/atomic/detail/string_ops.hpp>
#include <boost/atomic/detail/core_ops_cas_based.hpp>
#endif
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

struct core_arch_operations_gcc_x86_base
{
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = false;
static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void fence_before(memory_order order) BOOST_NOEXCEPT
{
if ((static_cast< unsigned int >(order) & static_cast< unsigned int >(memory_order_release)) != 0u)
__asm__ __volatile__ ("" ::: "memory");
}

static BOOST_FORCEINLINE void fence_after(memory_order order) BOOST_NOEXCEPT
{
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
__asm__ __volatile__ ("" ::: "memory");
}
};

template< std::size_t Size, bool Signed, bool Interprocess, typename Derived >
struct core_arch_operations_gcc_x86 :
public core_arch_operations_gcc_x86_base
{
typedef typename storage_traits< Size >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = Size;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = Size;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
if (order != memory_order_seq_cst)
{
fence_before(order);
storage = v;
fence_after(order);
}
else
{
Derived::exchange(storage, v, order);
}
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = storage;
fence_after(order);
return v;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
return Derived::fetch_add(storage, -v, order);
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
store(storage, (storage_type)0, order);
}
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 1u, Signed, Interprocess > :
public core_arch_operations_gcc_x86< 1u, Signed, Interprocess, core_arch_operations< 1u, Signed, Interprocess > >
{
typedef core_arch_operations_gcc_x86< 1u, Signed, Interprocess, core_arch_operations< 1u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;
typedef typename storage_traits< 4u >::type temp_storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"lock; xaddb %0, %1"
: "+q" (v), "+m" (storage)
:
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"xchgb %0, %1"
: "+q" (v), "+m" (storage)
:
: "memory"
);
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order, memory_order) BOOST_NOEXCEPT
{
storage_type previous = expected;
bool success;
#if defined(BOOST_ATOMIC_DETAIL_ASM_HAS_FLAG_OUTPUTS)
__asm__ __volatile__
(
"lock; cmpxchgb %3, %1"
: "+a" (previous), "+m" (storage), "=@ccz" (success)
: "q" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#else 
__asm__ __volatile__
(
"lock; cmpxchgb %3, %1\n\t"
"sete %2"
: "+a" (previous), "+m" (storage), "=q" (success)
: "q" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#endif 
expected = previous;
return success;
}

#define BOOST_ATOMIC_DETAIL_CAS_LOOP(op, argument, result)\
temp_storage_type new_val;\
__asm__ __volatile__\
(\
".align 16\n\t"\
"1: mov %[arg], %2\n\t"\
op " %%al, %b2\n\t"\
"lock; cmpxchgb %b2, %[storage]\n\t"\
"jne 1b"\
: [res] "+a" (result), [storage] "+m" (storage), "=&q" (new_val)\
: [arg] "ir" ((temp_storage_type)argument)\
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"\
)

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("andb", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("orb", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("xorb", v, res);
return res;
}

#undef BOOST_ATOMIC_DETAIL_CAS_LOOP
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 2u, Signed, Interprocess > :
public core_arch_operations_gcc_x86< 2u, Signed, Interprocess, core_arch_operations< 2u, Signed, Interprocess > >
{
typedef core_arch_operations_gcc_x86< 2u, Signed, Interprocess, core_arch_operations< 2u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;
typedef typename storage_traits< 4u >::type temp_storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"lock; xaddw %0, %1"
: "+q" (v), "+m" (storage)
:
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"xchgw %0, %1"
: "+q" (v), "+m" (storage)
:
: "memory"
);
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order, memory_order) BOOST_NOEXCEPT
{
storage_type previous = expected;
bool success;
#if defined(BOOST_ATOMIC_DETAIL_ASM_HAS_FLAG_OUTPUTS)
__asm__ __volatile__
(
"lock; cmpxchgw %3, %1"
: "+a" (previous), "+m" (storage), "=@ccz" (success)
: "q" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#else 
__asm__ __volatile__
(
"lock; cmpxchgw %3, %1\n\t"
"sete %2"
: "+a" (previous), "+m" (storage), "=q" (success)
: "q" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#endif 
expected = previous;
return success;
}

#define BOOST_ATOMIC_DETAIL_CAS_LOOP(op, argument, result)\
temp_storage_type new_val;\
__asm__ __volatile__\
(\
".align 16\n\t"\
"1: mov %[arg], %2\n\t"\
op " %%ax, %w2\n\t"\
"lock; cmpxchgw %w2, %[storage]\n\t"\
"jne 1b"\
: [res] "+a" (result), [storage] "+m" (storage), "=&q" (new_val)\
: [arg] "ir" ((temp_storage_type)argument)\
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"\
)

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("andw", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("orw", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("xorw", v, res);
return res;
}

#undef BOOST_ATOMIC_DETAIL_CAS_LOOP
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 4u, Signed, Interprocess > :
public core_arch_operations_gcc_x86< 4u, Signed, Interprocess, core_arch_operations< 4u, Signed, Interprocess > >
{
typedef core_arch_operations_gcc_x86< 4u, Signed, Interprocess, core_arch_operations< 4u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"lock; xaddl %0, %1"
: "+r" (v), "+m" (storage)
:
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"xchgl %0, %1"
: "+r" (v), "+m" (storage)
:
: "memory"
);
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order, memory_order) BOOST_NOEXCEPT
{
storage_type previous = expected;
bool success;
#if defined(BOOST_ATOMIC_DETAIL_ASM_HAS_FLAG_OUTPUTS)
__asm__ __volatile__
(
"lock; cmpxchgl %3, %1"
: "+a" (previous), "+m" (storage), "=@ccz" (success)
: "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#else 
__asm__ __volatile__
(
"lock; cmpxchgl %3, %1\n\t"
"sete %2"
: "+a" (previous), "+m" (storage), "=q" (success)
: "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#endif 
expected = previous;
return success;
}

#define BOOST_ATOMIC_DETAIL_CAS_LOOP(op, argument, result)\
storage_type new_val;\
__asm__ __volatile__\
(\
".align 16\n\t"\
"1: mov %[arg], %[new_val]\n\t"\
op " %%eax, %[new_val]\n\t"\
"lock; cmpxchgl %[new_val], %[storage]\n\t"\
"jne 1b"\
: [res] "+a" (result), [storage] "+m" (storage), [new_val] "=&r" (new_val)\
: [arg] "ir" (argument)\
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"\
)

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("andl", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("orl", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("xorl", v, res);
return res;
}

#undef BOOST_ATOMIC_DETAIL_CAS_LOOP
};

#if defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG8B)


template< bool Signed, bool Interprocess >
struct gcc_dcas_x86
{
typedef typename storage_traits< 8u >::type storage_type;
typedef uint32_t BOOST_ATOMIC_DETAIL_MAY_ALIAS aliasing_uint32_t;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 8u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 8u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = true;
static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
if (BOOST_LIKELY((((uintptr_t)&storage) & 0x00000007) == 0u))
{
#if defined(__SSE__)
typedef float xmm_t __attribute__((__vector_size__(16)));
xmm_t xmm_scratch;
__asm__ __volatile__
(
#if defined(__AVX__)
"vmovq %[value], %[xmm_scratch]\n\t"
"vmovq %[xmm_scratch], %[storage]\n\t"
#elif defined(__SSE2__)
"movq %[value], %[xmm_scratch]\n\t"
"movq %[xmm_scratch], %[storage]\n\t"
#else
"xorps %[xmm_scratch], %[xmm_scratch]\n\t"
"movlps %[value], %[xmm_scratch]\n\t"
"movlps %[xmm_scratch], %[storage]\n\t"
#endif
: [storage] "=m" (storage), [xmm_scratch] "=x" (xmm_scratch)
: [value] "m" (v)
: "memory"
);
#else
__asm__ __volatile__
(
"fildll %[value]\n\t"
"fistpll %[storage]\n\t"
: [storage] "=m" (storage)
: [value] "m" (v)
: "memory"
);
#endif
}
else
{
#if defined(BOOST_ATOMIC_DETAIL_X86_ASM_PRESERVE_EBX)
__asm__ __volatile__
(
"xchgl %%ebx, %%esi\n\t"
"movl %%eax, %%ebx\n\t"
"movl (%[dest]), %%eax\n\t"
"movl 4(%[dest]), %%edx\n\t"
".align 16\n\t"
"1: lock; cmpxchg8b (%[dest])\n\t"
"jne 1b\n\t"
"xchgl %%ebx, %%esi\n\t"
:
: "a" ((uint32_t)v), "c" ((uint32_t)(v >> 32)), [dest] "D" (&storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "edx", "memory"
);
#else 
__asm__ __volatile__
(
"movl %[dest_lo], %%eax\n\t"
"movl %[dest_hi], %%edx\n\t"
".align 16\n\t"
"1: lock; cmpxchg8b %[dest_lo]\n\t"
"jne 1b\n\t"
: [dest_lo] "=m" (storage), [dest_hi] "=m" (reinterpret_cast< volatile aliasing_uint32_t* >(&storage)[1])
: [value_lo] "b" ((uint32_t)v), "c" ((uint32_t)(v >> 32))
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "eax", "edx", "memory"
);
#endif 
}
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order) BOOST_NOEXCEPT
{
storage_type value;

if (BOOST_LIKELY((((uintptr_t)&storage) & 0x00000007) == 0u))
{
#if defined(__SSE__)
typedef float xmm_t __attribute__((__vector_size__(16)));
xmm_t xmm_scratch;
__asm__ __volatile__
(
#if defined(__AVX__)
"vmovq %[storage], %[xmm_scratch]\n\t"
"vmovq %[xmm_scratch], %[value]\n\t"
#elif defined(__SSE2__)
"movq %[storage], %[xmm_scratch]\n\t"
"movq %[xmm_scratch], %[value]\n\t"
#else
"xorps %[xmm_scratch], %[xmm_scratch]\n\t"
"movlps %[storage], %[xmm_scratch]\n\t"
"movlps %[xmm_scratch], %[value]\n\t"
#endif
: [value] "=m" (value), [xmm_scratch] "=x" (xmm_scratch)
: [storage] "m" (storage)
: "memory"
);
#else
__asm__ __volatile__
(
"fildll %[storage]\n\t"
"fistpll %[value]\n\t"
: [value] "=m" (value)
: [storage] "m" (storage)
: "memory"
);
#endif
}
else
{

#if defined(BOOST_ATOMIC_DETAIL_X86_NO_ASM_AX_DX_PAIRS)

uint32_t value_bits[2];

__asm__ __volatile__
(
"movl %%ebx, %%eax\n\t"
"movl %%ecx, %%edx\n\t"
"lock; cmpxchg8b %[storage]\n\t"
: "=&a" (value_bits[0]), "=&d" (value_bits[1])
: [storage] "m" (storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
BOOST_ATOMIC_DETAIL_MEMCPY(&value, value_bits, sizeof(value));

#else 

__asm__ __volatile__
(
"movl %%ebx, %%eax\n\t"
"movl %%ecx, %%edx\n\t"
"lock; cmpxchg8b %[storage]\n\t"
: "=&A" (value)
: [storage] "m" (storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

#endif 
}

return value;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order, memory_order) BOOST_NOEXCEPT
{
#if defined(__clang__)

storage_type old_expected = expected;
expected = __sync_val_compare_and_swap(&storage, old_expected, desired);
return expected == old_expected;

#elif defined(BOOST_ATOMIC_DETAIL_X86_ASM_PRESERVE_EBX)

bool success;

#if defined(BOOST_ATOMIC_DETAIL_ASM_HAS_FLAG_OUTPUTS)
__asm__ __volatile__
(
"xchgl %%ebx, %%esi\n\t"
"lock; cmpxchg8b (%[dest])\n\t"
"xchgl %%ebx, %%esi\n\t"
: "+A" (expected), [success] "=@ccz" (success)
: "S" ((uint32_t)desired), "c" ((uint32_t)(desired >> 32)), [dest] "D" (&storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#else 
__asm__ __volatile__
(
"xchgl %%ebx, %%esi\n\t"
"lock; cmpxchg8b (%[dest])\n\t"
"xchgl %%ebx, %%esi\n\t"
"sete %[success]\n\t"
: "+A" (expected), [success] "=qm" (success)
: "S" ((uint32_t)desired), "c" ((uint32_t)(desired >> 32)), [dest] "D" (&storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#endif 

return success;

#else 

bool success;

#if defined(BOOST_ATOMIC_DETAIL_ASM_HAS_FLAG_OUTPUTS)
__asm__ __volatile__
(
"lock; cmpxchg8b %[dest]\n\t"
: "+A" (expected), [dest] "+m" (storage), [success] "=@ccz" (success)
: "b" ((uint32_t)desired), "c" ((uint32_t)(desired >> 32))
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#else 
__asm__ __volatile__
(
"lock; cmpxchg8b %[dest]\n\t"
"sete %[success]\n\t"
: "+A" (expected), [dest] "+m" (storage), [success] "=qm" (success)
: "b" ((uint32_t)desired), "c" ((uint32_t)(desired >> 32))
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#endif 

return success;

#endif 
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
return compare_exchange_strong(storage, expected, desired, success_order, failure_order);
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
#if defined(BOOST_ATOMIC_DETAIL_X86_ASM_PRESERVE_EBX)
#if defined(BOOST_ATOMIC_DETAIL_X86_NO_ASM_AX_DX_PAIRS)

uint32_t old_bits[2];
__asm__ __volatile__
(
"xchgl %%ebx, %%esi\n\t"
"movl (%[dest]), %%eax\n\t"
"movl 4(%[dest]), %%edx\n\t"
".align 16\n\t"
"1: lock; cmpxchg8b (%[dest])\n\t"
"jne 1b\n\t"
"xchgl %%ebx, %%esi\n\t"
: "=a" (old_bits[0]), "=d" (old_bits[1])
: "S" ((uint32_t)v), "c" ((uint32_t)(v >> 32)), [dest] "D" (&storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

storage_type old_value;
BOOST_ATOMIC_DETAIL_MEMCPY(&old_value, old_bits, sizeof(old_value));
return old_value;

#else 

storage_type old_value;
__asm__ __volatile__
(
"xchgl %%ebx, %%esi\n\t"
"movl (%[dest]), %%eax\n\t"
"movl 4(%[dest]), %%edx\n\t"
".align 16\n\t"
"1: lock; cmpxchg8b (%[dest])\n\t"
"jne 1b\n\t"
"xchgl %%ebx, %%esi\n\t"
: "=A" (old_value)
: "S" ((uint32_t)v), "c" ((uint32_t)(v >> 32)), [dest] "D" (&storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
return old_value;

#endif 
#else 
#if defined(__MINGW32__) && ((__GNUC__+0) * 100 + (__GNUC_MINOR__+0)) < 407

uint32_t old_bits[2];
__asm__ __volatile__
(
"movl (%[dest]), %%eax\n\t"
"movl 4(%[dest]), %%edx\n\t"
".align 16\n\t"
"1: lock; cmpxchg8b (%[dest])\n\t"
"jne 1b\n\t"
: "=&a" (old_bits[0]), "=&d" (old_bits[1])
: "b" ((uint32_t)v), "c" ((uint32_t)(v >> 32)), [dest] "DS" (&storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

storage_type old_value;
BOOST_ATOMIC_DETAIL_MEMCPY(&old_value, old_bits, sizeof(old_value));
return old_value;

#elif defined(BOOST_ATOMIC_DETAIL_X86_NO_ASM_AX_DX_PAIRS)

uint32_t old_bits[2];
__asm__ __volatile__
(
"movl %[dest_lo], %%eax\n\t"
"movl %[dest_hi], %%edx\n\t"
".align 16\n\t"
"1: lock; cmpxchg8b %[dest_lo]\n\t"
"jne 1b\n\t"
: "=&a" (old_bits[0]), "=&d" (old_bits[1]), [dest_lo] "+m" (storage), [dest_hi] "+m" (reinterpret_cast< volatile aliasing_uint32_t* >(&storage)[1])
: "b" ((uint32_t)v), "c" ((uint32_t)(v >> 32))
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

storage_type old_value;
BOOST_ATOMIC_DETAIL_MEMCPY(&old_value, old_bits, sizeof(old_value));
return old_value;

#else 

storage_type old_value;
__asm__ __volatile__
(
"movl %[dest_lo], %%eax\n\t"
"movl %[dest_hi], %%edx\n\t"
".align 16\n\t"
"1: lock; cmpxchg8b %[dest_lo]\n\t"
"jne 1b\n\t"
: "=&A" (old_value), [dest_lo] "+m" (storage), [dest_hi] "+m" (reinterpret_cast< volatile aliasing_uint32_t* >(&storage)[1])
: "b" ((uint32_t)v), "c" ((uint32_t)(v >> 32))
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
return old_value;

#endif 
#endif 
}
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 8u, Signed, Interprocess > :
public core_operations_cas_based< gcc_dcas_x86< Signed, Interprocess > >
{
};

#elif defined(__x86_64__)

template< bool Signed, bool Interprocess >
struct core_arch_operations< 8u, Signed, Interprocess > :
public core_arch_operations_gcc_x86< 8u, Signed, Interprocess, core_arch_operations< 8u, Signed, Interprocess > >
{
typedef core_arch_operations_gcc_x86< 8u, Signed, Interprocess, core_arch_operations< 8u, Signed, Interprocess > > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"lock; xaddq %0, %1"
: "+r" (v), "+m" (storage)
:
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"xchgq %0, %1"
: "+r" (v), "+m" (storage)
:
: "memory"
);
return v;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order, memory_order) BOOST_NOEXCEPT
{
storage_type previous = expected;
bool success;
#if defined(BOOST_ATOMIC_DETAIL_ASM_HAS_FLAG_OUTPUTS)
__asm__ __volatile__
(
"lock; cmpxchgq %3, %1"
: "+a" (previous), "+m" (storage), "=@ccz" (success)
: "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#else 
__asm__ __volatile__
(
"lock; cmpxchgq %3, %1\n\t"
"sete %2"
: "+a" (previous), "+m" (storage), "=q" (success)
: "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#endif 
expected = previous;
return success;
}

#define BOOST_ATOMIC_DETAIL_CAS_LOOP(op, argument, result)\
storage_type new_val;\
__asm__ __volatile__\
(\
".align 16\n\t"\
"1: movq %[arg], %[new_val]\n\t"\
op " %%rax, %[new_val]\n\t"\
"lock; cmpxchgq %[new_val], %[storage]\n\t"\
"jne 1b"\
: [res] "+a" (result), [storage] "+m" (storage), [new_val] "=&r" (new_val)\
: [arg] "r" (argument)\
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"\
)

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("andq", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("orq", v, res);
return res;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
storage_type res = storage;
BOOST_ATOMIC_DETAIL_CAS_LOOP("xorq", v, res);
return res;
}

#undef BOOST_ATOMIC_DETAIL_CAS_LOOP
};

#endif

#if defined(BOOST_ATOMIC_DETAIL_X86_HAS_CMPXCHG16B)

template< bool Signed, bool Interprocess >
struct gcc_dcas_x86_64
{
typedef typename storage_traits< 16u >::type storage_type;
typedef uint64_t BOOST_ATOMIC_DETAIL_MAY_ALIAS aliasing_uint64_t;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 16u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 16u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = true;
static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
__asm__ __volatile__
(
"movq %[dest_lo], %%rax\n\t"
"movq %[dest_hi], %%rdx\n\t"
".align 16\n\t"
"1: lock; cmpxchg16b %[dest_lo]\n\t"
"jne 1b\n\t"
: [dest_lo] "=m" (storage), [dest_hi] "=m" (reinterpret_cast< volatile aliasing_uint64_t* >(&storage)[1])
: "b" (reinterpret_cast< const aliasing_uint64_t* >(&v)[0]), "c" (reinterpret_cast< const aliasing_uint64_t* >(&v)[1])
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "rax", "rdx", "memory"
);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order) BOOST_NOEXCEPT
{

#if defined(BOOST_ATOMIC_DETAIL_X86_NO_ASM_AX_DX_PAIRS)

uint64_t value_bits[2];

__asm__ __volatile__
(
"movq %%rbx, %%rax\n\t"
"movq %%rcx, %%rdx\n\t"
"lock; cmpxchg16b %[storage]\n\t"
: "=&a" (value_bits[0]), "=&d" (value_bits[1])
: [storage] "m" (storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

storage_type value;
BOOST_ATOMIC_DETAIL_MEMCPY(&value, value_bits, sizeof(value));
return value;

#else 

storage_type value;

__asm__ __volatile__
(
"movq %%rbx, %%rax\n\t"
"movq %%rcx, %%rdx\n\t"
"lock; cmpxchg16b %[storage]\n\t"
: "=&A" (value)
: [storage] "m" (storage)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

return value;

#endif 
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order, memory_order) BOOST_NOEXCEPT
{
#if defined(__clang__)

storage_type old_expected = expected;
expected = __sync_val_compare_and_swap(&storage, old_expected, desired);
return expected == old_expected;

#elif defined(BOOST_ATOMIC_DETAIL_X86_NO_ASM_AX_DX_PAIRS)

bool success;
__asm__ __volatile__
(
"lock; cmpxchg16b %[dest]\n\t"
"sete %[success]\n\t"
: [dest] "+m" (storage), "+a" (reinterpret_cast< aliasing_uint64_t* >(&expected)[0]), "+d" (reinterpret_cast< aliasing_uint64_t* >(&expected)[1]), [success] "=q" (success)
: "b" (reinterpret_cast< const aliasing_uint64_t* >(&desired)[0]), "c" (reinterpret_cast< const aliasing_uint64_t* >(&desired)[1])
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

return success;

#else 

bool success;

#if defined(BOOST_ATOMIC_DETAIL_ASM_HAS_FLAG_OUTPUTS)
__asm__ __volatile__
(
"lock; cmpxchg16b %[dest]\n\t"
: "+A" (expected), [dest] "+m" (storage), "=@ccz" (success)
: "b" (reinterpret_cast< const aliasing_uint64_t* >(&desired)[0]), "c" (reinterpret_cast< const aliasing_uint64_t* >(&desired)[1])
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#else 
__asm__ __volatile__
(
"lock; cmpxchg16b %[dest]\n\t"
"sete %[success]\n\t"
: "+A" (expected), [dest] "+m" (storage), [success] "=qm" (success)
: "b" (reinterpret_cast< const aliasing_uint64_t* >(&desired)[0]), "c" (reinterpret_cast< const aliasing_uint64_t* >(&desired)[1])
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);
#endif 

return success;

#endif 
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
return compare_exchange_strong(storage, expected, desired, success_order, failure_order);
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order) BOOST_NOEXCEPT
{
#if defined(BOOST_ATOMIC_DETAIL_X86_NO_ASM_AX_DX_PAIRS)
uint64_t old_bits[2];
__asm__ __volatile__
(
"movq %[dest_lo], %%rax\n\t"
"movq %[dest_hi], %%rdx\n\t"
".align 16\n\t"
"1: lock; cmpxchg16b %[dest_lo]\n\t"
"jne 1b\n\t"
: [dest_lo] "+m" (storage), [dest_hi] "+m" (reinterpret_cast< volatile aliasing_uint64_t* >(&storage)[1]), "=&a" (old_bits[0]), "=&d" (old_bits[1])
: "b" (reinterpret_cast< const aliasing_uint64_t* >(&v)[0]), "c" (reinterpret_cast< const aliasing_uint64_t* >(&v)[1])
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

storage_type old_value;
BOOST_ATOMIC_DETAIL_MEMCPY(&old_value, old_bits, sizeof(old_value));
return old_value;
#else 
storage_type old_value;
__asm__ __volatile__
(
"movq %[dest_lo], %%rax\n\t"
"movq %[dest_hi], %%rdx\n\t"
".align 16\n\t"
"1: lock; cmpxchg16b %[dest_lo]\n\t"
"jne 1b\n\t"
: "=&A" (old_value), [dest_lo] "+m" (storage), [dest_hi] "+m" (reinterpret_cast< volatile aliasing_uint64_t* >(&storage)[1])
: "b" (reinterpret_cast< const aliasing_uint64_t* >(&v)[0]), "c" (reinterpret_cast< const aliasing_uint64_t* >(&v)[1])
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC_COMMA "memory"
);

return old_value;
#endif 
}
};

template< bool Signed, bool Interprocess >
struct core_arch_operations< 16u, Signed, Interprocess > :
public core_operations_cas_based< gcc_dcas_x86_64< Signed, Interprocess > >
{
};

#endif 

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
