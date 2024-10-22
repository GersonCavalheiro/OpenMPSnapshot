

#if !defined(__TBB_machine_H) || defined(__TBB_machine_icc_generic_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#if ! __TBB_ICC_BUILTIN_ATOMICS_PRESENT
#error "Intel C++ Compiler of at least 12.0 version is needed to use ICC intrinsics port"
#endif

#define __TBB_machine_icc_generic_H

#if _MSC_VER
#include "msvc_ia32_common.h"
#else
#include "gcc_ia32_common.h"
#endif


#if __TBB_x86_32
#define __TBB_WORDSIZE 4
#else
#define __TBB_WORDSIZE 8
#endif
#define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE

#ifndef __TBB_compiler_fence
#if _MSC_VER
#pragma intrinsic(_ReadWriteBarrier)
#define __TBB_compiler_fence()    _ReadWriteBarrier()
#else
#define __TBB_compiler_fence()    __asm__ __volatile__("": : :"memory")
#endif
#endif

#ifndef __TBB_full_memory_fence
#if _MSC_VER
#pragma intrinsic(_mm_mfence)
#define __TBB_full_memory_fence() _mm_mfence()
#else
#define __TBB_full_memory_fence() __asm__ __volatile__("mfence": : :"memory")
#endif
#endif

#ifndef __TBB_control_consistency_helper
#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#endif

namespace tbb { namespace internal {
typedef enum memory_order {
memory_order_relaxed, memory_order_consume, memory_order_acquire,
memory_order_release, memory_order_acq_rel, memory_order_seq_cst
} memory_order;

namespace icc_intrinsics_port {
template <typename T>
T convert_argument(T value){
return value;
}
template <typename T>
void* convert_argument(T* value){
return (void*)value;
}
}
template <typename T, size_t S>
struct machine_load_store {
static T load_with_acquire ( const volatile T& location ) {
return __atomic_load_explicit(&location, memory_order_acquire);
}
static void store_with_release ( volatile T &location, T value ) {
__atomic_store_explicit(&location, icc_intrinsics_port::convert_argument(value), memory_order_release);
}
};

template <typename T, size_t S>
struct machine_load_store_relaxed {
static inline T load ( const T& location ) {
return __atomic_load_explicit(&location, memory_order_relaxed);
}
static inline void store (  T& location, T value ) {
__atomic_store_explicit(&location, icc_intrinsics_port::convert_argument(value), memory_order_relaxed);
}
};

template <typename T, size_t S>
struct machine_load_store_seq_cst {
static T load ( const volatile T& location ) {
return __atomic_load_explicit(&location, memory_order_seq_cst);
}

static void store ( volatile T &location, T value ) {
__atomic_store_explicit(&location, value, memory_order_seq_cst);
}
};

}} 

namespace tbb{ namespace internal { namespace icc_intrinsics_port{
typedef enum memory_order_map {
relaxed = memory_order_relaxed,
acquire = memory_order_acquire,
release = memory_order_release,
full_fence=  memory_order_seq_cst
} memory_order_map;
}}}

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T,M)                                                     \
inline T __TBB_machine_cmpswp##S##M( volatile void *ptr, T value, T comparand ) {               \
__atomic_compare_exchange_strong_explicit(                                                  \
(T*)ptr                                                                             \
,&comparand                                                                         \
,value                                                                              \
, tbb::internal::icc_intrinsics_port::M                                             \
, tbb::internal::icc_intrinsics_port::M);                                           \
return comparand;                                                                           \
}                                                                                               \
\
inline T __TBB_machine_fetchstore##S##M(volatile void *ptr, T value) {                          \
return __atomic_exchange_explicit((T*)ptr, value, tbb::internal::icc_intrinsics_port::M);   \
}                                                                                               \
\
inline T __TBB_machine_fetchadd##S##M(volatile void *ptr, T value) {                            \
return __atomic_fetch_add_explicit((T*)ptr, value, tbb::internal::icc_intrinsics_port::M);  \
}                                                                                               \

__TBB_MACHINE_DEFINE_ATOMICS(1,tbb::internal::int8_t, full_fence)
__TBB_MACHINE_DEFINE_ATOMICS(1,tbb::internal::int8_t, acquire)
__TBB_MACHINE_DEFINE_ATOMICS(1,tbb::internal::int8_t, release)
__TBB_MACHINE_DEFINE_ATOMICS(1,tbb::internal::int8_t, relaxed)

__TBB_MACHINE_DEFINE_ATOMICS(2,tbb::internal::int16_t, full_fence)
__TBB_MACHINE_DEFINE_ATOMICS(2,tbb::internal::int16_t, acquire)
__TBB_MACHINE_DEFINE_ATOMICS(2,tbb::internal::int16_t, release)
__TBB_MACHINE_DEFINE_ATOMICS(2,tbb::internal::int16_t, relaxed)

__TBB_MACHINE_DEFINE_ATOMICS(4,tbb::internal::int32_t, full_fence)
__TBB_MACHINE_DEFINE_ATOMICS(4,tbb::internal::int32_t, acquire)
__TBB_MACHINE_DEFINE_ATOMICS(4,tbb::internal::int32_t, release)
__TBB_MACHINE_DEFINE_ATOMICS(4,tbb::internal::int32_t, relaxed)

__TBB_MACHINE_DEFINE_ATOMICS(8,tbb::internal::int64_t, full_fence)
__TBB_MACHINE_DEFINE_ATOMICS(8,tbb::internal::int64_t, acquire)
__TBB_MACHINE_DEFINE_ATOMICS(8,tbb::internal::int64_t, release)
__TBB_MACHINE_DEFINE_ATOMICS(8,tbb::internal::int64_t, relaxed)


#undef __TBB_MACHINE_DEFINE_ATOMICS

#define __TBB_USE_FENCED_ATOMICS                            1

namespace tbb { namespace internal {
#if __TBB_FORCE_64BIT_ALIGNMENT_BROKEN
__TBB_MACHINE_DEFINE_LOAD8_GENERIC_FENCED(full_fence)
__TBB_MACHINE_DEFINE_STORE8_GENERIC_FENCED(full_fence)

__TBB_MACHINE_DEFINE_LOAD8_GENERIC_FENCED(acquire)
__TBB_MACHINE_DEFINE_STORE8_GENERIC_FENCED(release)

__TBB_MACHINE_DEFINE_LOAD8_GENERIC_FENCED(relaxed)
__TBB_MACHINE_DEFINE_STORE8_GENERIC_FENCED(relaxed)

template <typename T>
struct machine_load_store<T,8> {
static T load_with_acquire ( const volatile T& location ) {
if( tbb::internal::is_aligned(&location,8)) {
return __atomic_load_explicit(&location, memory_order_acquire);
} else {
return __TBB_machine_generic_load8acquire(&location);
}
}
static void store_with_release ( volatile T &location, T value ) {
if( tbb::internal::is_aligned(&location,8)) {
__atomic_store_explicit(&location, icc_intrinsics_port::convert_argument(value), memory_order_release);
} else {
return __TBB_machine_generic_store8release(&location,value);
}
}
};

template <typename T>
struct machine_load_store_relaxed<T,8> {
static T load( const volatile T& location ) {
if( tbb::internal::is_aligned(&location,8)) {
return __atomic_load_explicit(&location, memory_order_relaxed);
} else {
return __TBB_machine_generic_load8relaxed(&location);
}
}
static void store( volatile T &location, T value ) {
if( tbb::internal::is_aligned(&location,8)) {
__atomic_store_explicit(&location, icc_intrinsics_port::convert_argument(value), memory_order_relaxed);
} else {
return __TBB_machine_generic_store8relaxed(&location,value);
}
}
};

template <typename T >
struct machine_load_store_seq_cst<T,8> {
static T load ( const volatile T& location ) {
if( tbb::internal::is_aligned(&location,8)) {
return __atomic_load_explicit(&location, memory_order_seq_cst);
} else {
return __TBB_machine_generic_load8full_fence(&location);
}

}

static void store ( volatile T &location, T value ) {
if( tbb::internal::is_aligned(&location,8)) {
__atomic_store_explicit(&location, value, memory_order_seq_cst);
} else {
return __TBB_machine_generic_store8full_fence(&location,value);
}

}
};

#endif
}} 
template <typename T>
inline void __TBB_machine_OR( T *operand, T addend ) {
__atomic_fetch_or_explicit(operand, addend, tbb::internal::memory_order_seq_cst);
}

template <typename T>
inline void __TBB_machine_AND( T *operand, T addend ) {
__atomic_fetch_and_explicit(operand, addend, tbb::internal::memory_order_seq_cst);
}

