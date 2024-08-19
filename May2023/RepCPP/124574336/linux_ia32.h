

#if !defined(__TBB_machine_H) || defined(__TBB_machine_linux_ia32_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_linux_ia32_H

#include <stdint.h>
#include "gcc_ia32_common.h"

#define __TBB_WORDSIZE 4
#define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE

#define __TBB_compiler_fence() __asm__ __volatile__("": : :"memory")
#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#define __TBB_acquire_consistency_helper() __TBB_compiler_fence()
#define __TBB_release_consistency_helper() __TBB_compiler_fence()
#define __TBB_full_memory_fence()          __asm__ __volatile__("mfence": : :"memory")

#if __TBB_ICC_ASM_VOLATILE_BROKEN
#define __TBB_VOLATILE
#else
#define __TBB_VOLATILE volatile
#endif

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T,X,R)                                        \
static inline T __TBB_machine_cmpswp##S (volatile void *ptr, T value, T comparand )  \
{                                                                                    \
T result;                                                                        \
\
__asm__ __volatile__("lock\ncmpxchg" X " %2,%1"                                  \
: "=a"(result), "=m"(*(__TBB_VOLATILE T*)ptr)              \
: "q"(value), "0"(comparand), "m"(*(__TBB_VOLATILE T*)ptr) \
: "memory");                                               \
return result;                                                                   \
}                                                                                    \
\
static inline T __TBB_machine_fetchadd##S(volatile void *ptr, T addend)              \
{                                                                                    \
T result;                                                                        \
__asm__ __volatile__("lock\nxadd" X " %0,%1"                                     \
: R (result), "=m"(*(__TBB_VOLATILE T*)ptr)                \
: "0"(addend), "m"(*(__TBB_VOLATILE T*)ptr)                \
: "memory");                                               \
return result;                                                                   \
}                                                                                    \
\
static inline  T __TBB_machine_fetchstore##S(volatile void *ptr, T value)            \
{                                                                                    \
T result;                                                                        \
__asm__ __volatile__("lock\nxchg" X " %0,%1"                                     \
: R (result), "=m"(*(__TBB_VOLATILE T*)ptr)                \
: "0"(value), "m"(*(__TBB_VOLATILE T*)ptr)                 \
: "memory");                                               \
return result;                                                                   \
}                                                                                    \

__TBB_MACHINE_DEFINE_ATOMICS(1,int8_t,"","=q")
__TBB_MACHINE_DEFINE_ATOMICS(2,int16_t,"","=r")
__TBB_MACHINE_DEFINE_ATOMICS(4,int32_t,"l","=r")

#if __INTEL_COMPILER
#pragma warning( push )
#pragma warning( disable: 998 )
#endif

#if __TBB_GCC_CAS8_BUILTIN_INLINING_BROKEN
#define  __TBB_IA32_CAS8_NOINLINE  __attribute__ ((noinline))
#else
#define  __TBB_IA32_CAS8_NOINLINE
#endif

static inline __TBB_IA32_CAS8_NOINLINE int64_t __TBB_machine_cmpswp8 (volatile void *ptr, int64_t value, int64_t comparand )  {
#if (__TBB_GCC_BUILTIN_ATOMICS_PRESENT || (__TBB_GCC_VERSION >= 40102)) && !__TBB_GCC_64BIT_ATOMIC_BUILTINS_BROKEN
return __sync_val_compare_and_swap( reinterpret_cast<volatile int64_t*>(ptr), comparand, value );
#else 
int64_t result;
union {
int64_t i64;
int32_t i32[2];
};
i64 = value;
#if __PIC__

int32_t tmp;
__asm__ __volatile__ (
"movl  %%ebx,%2\n\t"
"movl  %5,%%ebx\n\t"
#if __GNUC__==3
"lock\n\t cmpxchg8b %1\n\t"
#else
"lock\n\t cmpxchg8b (%3)\n\t"
#endif
"movl  %2,%%ebx"
: "=A"(result)
, "=m"(*(__TBB_VOLATILE int64_t *)ptr)
, "=m"(tmp)
#if __GNUC__==3
: "m"(*(__TBB_VOLATILE int64_t *)ptr)
#else
: "SD"(ptr)
#endif
, "0"(comparand)
, "m"(i32[0]), "c"(i32[1])
: "memory"
#if __INTEL_COMPILER
,"ebx"
#endif
);
#else 
__asm__ __volatile__ (
"lock\n\t cmpxchg8b %1\n\t"
: "=A"(result), "=m"(*(__TBB_VOLATILE int64_t *)ptr)
: "m"(*(__TBB_VOLATILE int64_t *)ptr)
, "0"(comparand)
, "b"(i32[0]), "c"(i32[1])
: "memory"
);
#endif 
return result;
#endif 
}

#undef __TBB_IA32_CAS8_NOINLINE

#if __INTEL_COMPILER
#pragma warning( pop )
#endif 

static inline void __TBB_machine_or( volatile void *ptr, uint32_t addend ) {
__asm__ __volatile__("lock\norl %1,%0" : "=m"(*(__TBB_VOLATILE uint32_t *)ptr) : "r"(addend), "m"(*(__TBB_VOLATILE uint32_t *)ptr) : "memory");
}

static inline void __TBB_machine_and( volatile void *ptr, uint32_t addend ) {
__asm__ __volatile__("lock\nandl %1,%0" : "=m"(*(__TBB_VOLATILE uint32_t *)ptr) : "r"(addend), "m"(*(__TBB_VOLATILE uint32_t *)ptr) : "memory");
}


#if __clang__
#define __TBB_fildq  "fildll"
#define __TBB_fistpq "fistpll"
#else
#define __TBB_fildq  "fildq"
#define __TBB_fistpq "fistpq"
#endif

static inline int64_t __TBB_machine_aligned_load8 (const volatile void *ptr) {
__TBB_ASSERT(tbb::internal::is_aligned(ptr,8),"__TBB_machine_aligned_load8 should be used with 8 byte aligned locations only \n");
int64_t result;
__asm__ __volatile__ ( __TBB_fildq  " %1\n\t"
__TBB_fistpq " %0" :  "=m"(result) : "m"(*(const __TBB_VOLATILE uint64_t*)ptr) : "memory" );
return result;
}

static inline void __TBB_machine_aligned_store8 (volatile void *ptr, int64_t value ) {
__TBB_ASSERT(tbb::internal::is_aligned(ptr,8),"__TBB_machine_aligned_store8 should be used with 8 byte aligned locations only \n");
__asm__ __volatile__ ( __TBB_fildq  " %1\n\t"
__TBB_fistpq " %0" :  "=m"(*(__TBB_VOLATILE int64_t*)ptr) : "m"(value) : "memory" );
}

static inline int64_t __TBB_machine_load8 (const volatile void *ptr) {
#if __TBB_FORCE_64BIT_ALIGNMENT_BROKEN
if( tbb::internal::is_aligned(ptr,8)) {
#endif
return __TBB_machine_aligned_load8(ptr);
#if __TBB_FORCE_64BIT_ALIGNMENT_BROKEN
} else {
return __TBB_machine_cmpswp8(const_cast<void*>(ptr),0,0);
}
#endif
}


extern "C" void __TBB_machine_store8_slow( volatile void *ptr, int64_t value );
extern "C" void __TBB_machine_store8_slow_perf_warning( volatile void *ptr );

static inline void __TBB_machine_store8(volatile void *ptr, int64_t value) {
#if __TBB_FORCE_64BIT_ALIGNMENT_BROKEN
if( tbb::internal::is_aligned(ptr,8)) {
#endif
__TBB_machine_aligned_store8(ptr,value);
#if __TBB_FORCE_64BIT_ALIGNMENT_BROKEN
} else {
#if TBB_USE_PERFORMANCE_WARNINGS
__TBB_machine_store8_slow_perf_warning(ptr);
#endif 
__TBB_machine_store8_slow(ptr,value);
}
#endif
}

#define __TBB_AtomicOR(P,V) __TBB_machine_or(P,V)
#define __TBB_AtomicAND(P,V) __TBB_machine_and(P,V)

#define __TBB_USE_GENERIC_DWORD_FETCH_ADD                   1
#define __TBB_USE_GENERIC_DWORD_FETCH_STORE                 1
#define __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE           1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

