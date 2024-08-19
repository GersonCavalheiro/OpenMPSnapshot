


#ifndef BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_ARM_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_ARM_HPP_INCLUDED_

#include <cstddef>
#include <boost/cstdint.hpp>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/storage_traits.hpp>
#include <boost/atomic/detail/integral_conversions.hpp>
#include <boost/atomic/detail/core_arch_operations_fwd.hpp>
#include <boost/atomic/detail/ops_gcc_arm_common.hpp>
#include <boost/atomic/detail/gcc_arm_asm_common.hpp>
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
public core_arch_operations_gcc_arm_base
{
typedef typename storage_traits< 4u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 4u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 4u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage = v;
fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = storage;
fence_after(order);
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage_type original;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex %[original], %[storage]\n\t"          
"strex %[tmp], %[value], %[storage]\n\t"     
"teq   %[tmp], #0\n\t"                       
"bne   1b\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [tmp] "=&l" (tmp), [original] "=&r" (original), [storage] "+Q" (storage)
: [value] "r" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
bool success = false;
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
uint32_t tmp;
#endif
storage_type original;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"ldrex   %[original], %[storage]\n\t"             
"cmp     %[original], %[expected]\n\t"            
"itt     eq\n\t"                                  
"strexeq %[success], %[desired], %[storage]\n\t"  
"eoreq   %[success], %[success], #1\n\t"          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),
[success] "+r" (success),
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
[tmp] "=&l" (tmp),
#endif
[storage] "+Q" (storage)
: [expected] "Ir" (expected),
[desired] "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = original;
return success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
bool success = false;
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
uint32_t tmp;
#endif
storage_type original;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"             
"cmp     %[original], %[expected]\n\t"            
"bne     2f\n\t"                                  
"strex   %[success], %[desired], %[storage]\n\t"  
"eors    %[success], %[success], #1\n\t"          
"beq     1b\n\t"                                  
"2:\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),
[success] "+r" (success),
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
[tmp] "=&l" (tmp),
#endif
[storage] "+Q" (storage)
: [expected] "Ir" (expected),
[desired] "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = original;
return success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"add     %[result], %[original], %[value]\n\t"  
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"sub     %[result], %[original], %[value]\n\t"  
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"and     %[result], %[original], %[value]\n\t"  
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"orr     %[result], %[original], %[value]\n\t"  
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"eor     %[result], %[original], %[value]\n\t"  
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
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

#if defined(BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXB_STREXB)

template< bool Signed, bool Interprocess >
struct core_arch_operations< 1u, Signed, Interprocess > :
public core_arch_operations_gcc_arm_base
{
typedef typename storage_traits< 1u >::type storage_type;
typedef typename storage_traits< 4u >::type extended_storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 1u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 1u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage = v;
fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = storage;
fence_after(order);
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
extended_storage_type original;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexb %[original], %[storage]\n\t"          
"strexb %[tmp], %[value], %[storage]\n\t"     
"teq    %[tmp], #0\n\t"                       
"bne    1b\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [tmp] "=&l" (tmp), [original] "=&r" (original), [storage] "+Q" (storage)
: [value] "r" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
bool success = false;
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
uint32_t tmp;
#endif
extended_storage_type original;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"ldrexb   %[original], %[storage]\n\t"             
"cmp      %[original], %[expected]\n\t"            
"itt      eq\n\t"                                  
"strexbeq %[success], %[desired], %[storage]\n\t"  
"eoreq    %[success], %[success], #1\n\t"          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),
[success] "+r" (success),
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
[tmp] "=&l" (tmp),
#endif
[storage] "+Q" (storage)
: [expected] "Ir" (atomics::detail::zero_extend< extended_storage_type >(expected)),
[desired] "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = static_cast< storage_type >(original);
return success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
bool success = false;
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
uint32_t tmp;
#endif
extended_storage_type original;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexb   %[original], %[storage]\n\t"             
"cmp      %[original], %[expected]\n\t"            
"bne      2f\n\t"                                  
"strexb   %[success], %[desired], %[storage]\n\t"  
"eors     %[success], %[success], #1\n\t"          
"beq      1b\n\t"                                  
"2:\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),
[success] "+r" (success),
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
[tmp] "=&l" (tmp),
#endif
[storage] "+Q" (storage)
: [expected] "Ir" (atomics::detail::zero_extend< extended_storage_type >(expected)),
[desired] "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = static_cast< storage_type >(original);
return success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexb   %[original], %[storage]\n\t"           
"add      %[result], %[original], %[value]\n\t"  
"strexb   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexb   %[original], %[storage]\n\t"           
"sub      %[result], %[original], %[value]\n\t"  
"strexb   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexb   %[original], %[storage]\n\t"           
"and      %[result], %[original], %[value]\n\t"  
"strexb   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexb   %[original], %[storage]\n\t"           
"orr      %[result], %[original], %[value]\n\t"  
"strexb   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexb   %[original], %[storage]\n\t"           
"eor      %[result], %[original], %[value]\n\t"  
"strexb   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
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
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"add     %[result], %[original], %[value]\n\t"  
"uxtb    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"sub     %[result], %[original], %[value]\n\t"  
"uxtb    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
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
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"add     %[result], %[original], %[value]\n\t"  
"sxtb    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"sub     %[result], %[original], %[value]\n\t"  
"sxtb    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};

#endif 

#if defined(BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXH_STREXH)

template< bool Signed, bool Interprocess >
struct core_arch_operations< 2u, Signed, Interprocess > :
public core_arch_operations_gcc_arm_base
{
typedef typename storage_traits< 2u >::type storage_type;
typedef typename storage_traits< 4u >::type extended_storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 2u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 2u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage = v;
fence_after_store(order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type v = storage;
fence_after(order);
return v;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
extended_storage_type original;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexh %[original], %[storage]\n\t"          
"strexh %[tmp], %[value], %[storage]\n\t"     
"teq    %[tmp], #0\n\t"                       
"bne    1b\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [tmp] "=&l" (tmp), [original] "=&r" (original), [storage] "+Q" (storage)
: [value] "r" (v)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
bool success = false;
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
uint32_t tmp;
#endif
extended_storage_type original;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"ldrexh   %[original], %[storage]\n\t"             
"cmp      %[original], %[expected]\n\t"            
"itt      eq\n\t"                                  
"strexheq %[success], %[desired], %[storage]\n\t"  
"eoreq    %[success], %[success], #1\n\t"          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),
[success] "+r" (success),
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
[tmp] "=&l" (tmp),
#endif
[storage] "+Q" (storage)
: [expected] "Ir" (atomics::detail::zero_extend< extended_storage_type >(expected)),
[desired] "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = static_cast< storage_type >(original);
return success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
bool success = false;
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
uint32_t tmp;
#endif
extended_storage_type original;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexh   %[original], %[storage]\n\t"             
"cmp      %[original], %[expected]\n\t"            
"bne      2f\n\t"                                  
"strexh   %[success], %[desired], %[storage]\n\t"  
"eors     %[success], %[success], #1\n\t"          
"beq      1b\n\t"                                  
"2:\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),
[success] "+r" (success),
#if !defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
[tmp] "=&l" (tmp),
#endif
[storage] "+Q" (storage)
: [expected] "Ir" (atomics::detail::zero_extend< extended_storage_type >(expected)),
[desired] "r" (desired)
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = static_cast< storage_type >(original);
return success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexh   %[original], %[storage]\n\t"           
"add      %[result], %[original], %[value]\n\t"  
"strexh   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexh   %[original], %[storage]\n\t"           
"sub      %[result], %[original], %[value]\n\t"  
"strexh   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexh   %[original], %[storage]\n\t"           
"and      %[result], %[original], %[value]\n\t"  
"strexh   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexh   %[original], %[storage]\n\t"           
"orr      %[result], %[original], %[value]\n\t"  
"strexh   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
uint32_t tmp;
extended_storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrexh   %[original], %[storage]\n\t"           
"eor      %[result], %[original], %[value]\n\t"  
"strexh   %[tmp], %[result], %[storage]\n\t"     
"teq      %[tmp], #0\n\t"                        
"bne      1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return static_cast< storage_type >(original);
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
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"add     %[result], %[original], %[value]\n\t"  
"uxth    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"sub     %[result], %[original], %[value]\n\t"  
"uxth    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
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
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"add     %[result], %[original], %[value]\n\t"  
"sxth    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
base_type::fence_before(order);
uint32_t tmp;
storage_type original, result;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%[tmp])
"1:\n\t"
"ldrex   %[original], %[storage]\n\t"           
"sub     %[result], %[original], %[value]\n\t"  
"sxth    %[result], %[result]\n\t"              
"strex   %[tmp], %[result], %[storage]\n\t"     
"teq     %[tmp], #0\n\t"                        
"bne     1b\n\t"                                
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%[tmp])
: [original] "=&r" (original),  
[result] "=&r" (result),      
[tmp] "=&l" (tmp),            
[storage] "+Q" (storage)      
: [value] "Ir" (v)              
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};

#endif 

#if defined(BOOST_ATOMIC_DETAIL_ARM_HAS_LDREXD_STREXD)



template< bool Signed, bool Interprocess >
struct core_arch_operations< 8u, Signed, Interprocess > :
public core_arch_operations_gcc_arm_base
{
typedef typename storage_traits< 8u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 8u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 8u;
static BOOST_CONSTEXPR_OR_CONST bool is_signed = Signed;
static BOOST_CONSTEXPR_OR_CONST bool is_interprocess = Interprocess;

static BOOST_FORCEINLINE void store(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
exchange(storage, v, order);
}

static BOOST_FORCEINLINE storage_type load(storage_type const volatile& storage, memory_order order) BOOST_NOEXCEPT
{
storage_type original;
#if defined(BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_UNUSED)
__asm__ __volatile__
(
"ldrexd %0, %H0, %1\n\t"
: "=&r" (original)   
: "Q" (storage)      
);
#else
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"ldrexd %1, %H1, %2\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original)   
: "Q" (storage)      
);
#endif
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type exchange(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage_type original;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"1:\n\t"
"ldrexd %1, %H1, %2\n\t"        
"strexd %0, %3, %H3, %2\n\t"    
"teq    %0, #0\n\t"               
"bne    1b\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"+Q" (storage)     
: "r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
storage_type original;
bool success = false;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"ldrexd   %1, %H1, %3\n\t"               
"cmp      %1, %4\n\t"                    
"it       eq\n\t"                        
"cmpeq    %H1, %H4\n\t"                  
"bne      1f\n\t"
"strexd   %2, %5, %H5, %3\n\t"           
"eor      %2, %2, #1\n\t"                
"1:\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"+r" (success),    
"+Q" (storage)     
: "r" (expected),    
"r" (desired)      
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = original;
return success;
}

static BOOST_FORCEINLINE bool compare_exchange_strong(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
storage_type original;
bool success = false;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"1:\n\t"
"ldrexd   %1, %H1, %3\n\t"               
"cmp      %1, %4\n\t"                    
"it       eq\n\t"                        
"cmpeq    %H1, %H4\n\t"                  
"bne      2f\n\t"
"strexd   %2, %5, %H5, %3\n\t"           
"eors     %2, %2, #1\n\t"                
"beq      1b\n\t"                        
"2:\n\t"
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"+r" (success),    
"+Q" (storage)     
: "r" (expected),    
"r" (desired)      
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
expected = original;
return success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage_type original, result;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"1:\n\t"
"ldrexd  %1, %H1, %3\n\t"                 
"adds   " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_LO(2) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_LO(1) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_LO(4) "\n\t" 
"adc    " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_HI(2) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_HI(1) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_HI(4) "\n\t"
"strexd  %0, %2, %H2, %3\n\t"             
"teq     %0, #0\n\t"                      
"bne     1b\n\t"                          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"=&r" (result),    
"+Q" (storage)     
: "r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage_type original, result;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"1:\n\t"
"ldrexd  %1, %H1, %3\n\t"                 
"subs   " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_LO(2) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_LO(1) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_LO(4) "\n\t" 
"sbc    " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_HI(2) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_HI(1) ", " BOOST_ATOMIC_DETAIL_ARM_ASM_ARG_HI(4) "\n\t"
"strexd  %0, %2, %H2, %3\n\t"             
"teq     %0, #0\n\t"                      
"bne     1b\n\t"                          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"=&r" (result),    
"+Q" (storage)     
: "r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage_type original, result;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"1:\n\t"
"ldrexd  %1, %H1, %3\n\t"                 
"and     %2, %1, %4\n\t"                  
"and     %H2, %H1, %H4\n\t"
"strexd  %0, %2, %H2, %3\n\t"             
"teq     %0, #0\n\t"                      
"bne     1b\n\t"                          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"=&r" (result),    
"+Q" (storage)     
: "r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage_type original, result;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"1:\n\t"
"ldrexd  %1, %H1, %3\n\t"                 
"orr     %2, %1, %4\n\t"                  
"orr     %H2, %H1, %H4\n\t"
"strexd  %0, %2, %H2, %3\n\t"             
"teq     %0, #0\n\t"                      
"bne     1b\n\t"                          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"=&r" (result),    
"+Q" (storage)     
: "r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
fence_before(order);
storage_type original, result;
uint32_t tmp;
__asm__ __volatile__
(
BOOST_ATOMIC_DETAIL_ARM_ASM_START(%0)
"1:\n\t"
"ldrexd  %1, %H1, %3\n\t"                 
"eor     %2, %1, %4\n\t"                  
"eor     %H2, %H1, %H4\n\t"
"strexd  %0, %2, %H2, %3\n\t"             
"teq     %0, #0\n\t"                      
"bne     1b\n\t"                          
BOOST_ATOMIC_DETAIL_ARM_ASM_END(%0)
: BOOST_ATOMIC_DETAIL_ARM_ASM_TMPREG_CONSTRAINT(tmp), 
"=&r" (original),  
"=&r" (result),    
"+Q" (storage)     
: "r" (v)            
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
