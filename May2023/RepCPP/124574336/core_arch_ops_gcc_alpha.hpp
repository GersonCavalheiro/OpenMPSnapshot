


#ifndef BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_ALPHA_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_ARCH_OPS_GCC_ALPHA_HPP_INCLUDED_

#include <cstddef>
#include <boost/memory_order.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/storage_traits.hpp>
#include <boost/atomic/detail/core_arch_operations_fwd.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {





struct core_arch_operations_gcc_alpha_base
{
static BOOST_CONSTEXPR_OR_CONST bool full_cas_based = false;
static BOOST_CONSTEXPR_OR_CONST bool is_always_lock_free = true;

static BOOST_FORCEINLINE void fence_before(memory_order order) BOOST_NOEXCEPT
{
if ((static_cast< unsigned int >(order) & static_cast< unsigned int >(memory_order_release)) != 0u)
__asm__ __volatile__ ("mb" ::: "memory");
}

static BOOST_FORCEINLINE void fence_after(memory_order order) BOOST_NOEXCEPT
{
if ((static_cast< unsigned int >(order) & (static_cast< unsigned int >(memory_order_consume) | static_cast< unsigned int >(memory_order_acquire))) != 0u)
__asm__ __volatile__ ("mb" ::: "memory");
}

static BOOST_FORCEINLINE void fence_after_store(memory_order order) BOOST_NOEXCEPT
{
if (order == memory_order_seq_cst)
__asm__ __volatile__ ("mb" ::: "memory");
}
};


template< bool Signed, bool Interprocess >
struct core_arch_operations< 4u, Signed, Interprocess > :
public core_arch_operations_gcc_alpha_base
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
storage_type original, tmp;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"mov %3, %1\n\t"
"ldl_l %0, %2\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (tmp)        
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
int success;
storage_type current;
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %2, %4\n\t"                
"cmpeq %2, %0, %3\n\t"            
"mov %2, %0\n\t"                  
"beq %3, 2f\n\t"                  
"stl_c %1, %4\n\t"                
"mov %1, %3\n\t"                  
"2:\n\t"
: "+r" (expected),   
"+r" (desired),    
"=&r" (current),   
"=&r" (success)    
: "m" (storage)      
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
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
storage_type current, tmp;
fence_before(success_order);
__asm__ __volatile__
(
"1:\n\t"
"mov %5, %1\n\t"                  
"ldl_l %2, %4\n\t"                
"cmpeq %2, %0, %3\n\t"            
"mov %2, %0\n\t"                  
"beq %3, 2f\n\t"                  
"stl_c %1, %4\n\t"                
"beq %1, 3f\n\t"                  
"mov %1, %3\n\t"                  
"2:\n\t"

".subsection 2\n\t"
"3: br 1b\n\t"
".previous\n\t"

: "+r" (expected),   
"=&r" (tmp),       
"=&r" (current),   
"=&r" (success)    
: "m" (storage),     
"r" (desired)      
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"addl %0, %3, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"subl %0, %3, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"and %0, %3, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"bis %0, %3, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"xor %0, %3, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
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
store(storage, 0, order);
}
};


template< bool Interprocess >
struct core_arch_operations< 1u, false, Interprocess > :
public core_arch_operations< 4u, false, Interprocess >
{
typedef core_arch_operations< 4u, false, Interprocess > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"addl %0, %3, %1\n\t"
"zapnot %1, 1, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"subl %0, %3, %1\n\t"
"zapnot %1, 1, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
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
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"addl %0, %3, %1\n\t"
"sextb %1, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"subl %0, %3, %1\n\t"
"sextb %1, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};


template< bool Interprocess >
struct core_arch_operations< 2u, false, Interprocess > :
public core_arch_operations< 4u, false, Interprocess >
{
typedef core_arch_operations< 4u, false, Interprocess > base_type;
typedef typename base_type::storage_type storage_type;

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"addl %0, %3, %1\n\t"
"zapnot %1, 3, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"subl %0, %3, %1\n\t"
"zapnot %1, 3, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
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
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"addl %0, %3, %1\n\t"
"sextw %1, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
base_type::fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldl_l %0, %2\n\t"
"subl %0, %3, %1\n\t"
"sextw %1, %1\n\t"
"stl_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
base_type::fence_after(order);
return original;
}
};


template< bool Signed, bool Interprocess >
struct core_arch_operations< 8u, Signed, Interprocess > :
public core_arch_operations_gcc_alpha_base
{
typedef typename storage_traits< 8u >::type storage_type;

static BOOST_CONSTEXPR_OR_CONST std::size_t storage_size = 8u;
static BOOST_CONSTEXPR_OR_CONST std::size_t storage_alignment = 8u;
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
storage_type original, tmp;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"mov %3, %1\n\t"
"ldq_l %0, %2\n\t"
"stq_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (tmp)        
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE bool compare_exchange_weak(
storage_type volatile& storage, storage_type& expected, storage_type desired, memory_order success_order, memory_order failure_order) BOOST_NOEXCEPT
{
fence_before(success_order);
int success;
storage_type current;
__asm__ __volatile__
(
"1:\n\t"
"ldq_l %2, %4\n\t"                
"cmpeq %2, %0, %3\n\t"            
"mov %2, %0\n\t"                  
"beq %3, 2f\n\t"                  
"stq_c %1, %4\n\t"                
"mov %1, %3\n\t"                  
"2:\n\t"
: "+r" (expected),   
"+r" (desired),    
"=&r" (current),   
"=&r" (success)    
: "m" (storage)      
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
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
storage_type current, tmp;
fence_before(success_order);
__asm__ __volatile__
(
"1:\n\t"
"mov %5, %1\n\t"                  
"ldq_l %2, %4\n\t"                
"cmpeq %2, %0, %3\n\t"            
"mov %2, %0\n\t"                  
"beq %3, 2f\n\t"                  
"stq_c %1, %4\n\t"                
"beq %1, 3f\n\t"                  
"mov %1, %3\n\t"                  
"2:\n\t"

".subsection 2\n\t"
"3: br 1b\n\t"
".previous\n\t"

: "+r" (expected),   
"=&r" (tmp),       
"=&r" (current),   
"=&r" (success)    
: "m" (storage),     
"r" (desired)      
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
if (success)
fence_after(success_order);
else
fence_after(failure_order);
return !!success;
}

static BOOST_FORCEINLINE storage_type fetch_add(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldq_l %0, %2\n\t"
"addq %0, %3, %1\n\t"
"stq_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_sub(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldq_l %0, %2\n\t"
"subq %0, %3, %1\n\t"
"stq_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_and(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldq_l %0, %2\n\t"
"and %0, %3, %1\n\t"
"stq_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_or(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldq_l %0, %2\n\t"
"bis %0, %3, %1\n\t"
"stq_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
: BOOST_ATOMIC_DETAIL_ASM_CLOBBER_CC
);
fence_after(order);
return original;
}

static BOOST_FORCEINLINE storage_type fetch_xor(storage_type volatile& storage, storage_type v, memory_order order) BOOST_NOEXCEPT
{
storage_type original, modified;
fence_before(order);
__asm__ __volatile__
(
"1:\n\t"
"ldq_l %0, %2\n\t"
"xor %0, %3, %1\n\t"
"stq_c %1, %2\n\t"
"beq %1, 2f\n\t"

".subsection 2\n\t"
"2: br 1b\n\t"
".previous\n\t"

: "=&r" (original),  
"=&r" (modified)   
: "m" (storage),     
"r" (v)            
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

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
