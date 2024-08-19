

#ifndef __TBB_machine_H
#define __TBB_machine_H



#include "tbb_stddef.h"

namespace tbb {
namespace internal { 

\
const int64_t anyvalue = 2305843009213693951LL;                                      \
return __TBB_machine_cmpswp8##M(const_cast<volatile void *>(ptr),anyvalue,anyvalue); \
}                                                                                        \

#define __TBB_ENDIAN_UNSUPPORTED -1
#define __TBB_ENDIAN_LITTLE       0
#define __TBB_ENDIAN_BIG          1
#define __TBB_ENDIAN_DETECT       2

#if _WIN32||_WIN64

#ifdef _MANAGED
#pragma managed(push, off)
#endif

#if __MINGW64__ || __MINGW32__
extern "C" __declspec(dllimport) int __stdcall SwitchToThread( void );
#define __TBB_Yield()  SwitchToThread()
#if (TBB_USE_GCC_BUILTINS && __TBB_GCC_BUILTIN_ATOMICS_PRESENT)
#include "machine/gcc_generic.h"
#elif __MINGW64__
#include "machine/linux_intel64.h"
#elif __MINGW32__
#include "machine/linux_ia32.h"
#endif
#elif (TBB_USE_ICC_BUILTINS && __TBB_ICC_BUILTIN_ATOMICS_PRESENT)
#include "machine/icc_generic.h"
#elif defined(_M_IX86) && !defined(__TBB_WIN32_USE_CL_BUILTINS)
#include "machine/windows_ia32.h"
#elif defined(_M_X64)
#include "machine/windows_intel64.h"
#elif defined(_M_ARM) || defined(__TBB_WIN32_USE_CL_BUILTINS)
#include "machine/msvc_armv7.h"
#endif

#ifdef _MANAGED
#pragma managed(pop)
#endif

#elif __TBB_DEFINE_MIC

#include "machine/mic_common.h"
#if (TBB_USE_ICC_BUILTINS && __TBB_ICC_BUILTIN_ATOMICS_PRESENT)
#include "machine/icc_generic.h"
#else
#include "machine/linux_intel64.h"
#endif

#elif __linux__ || __FreeBSD__ || __NetBSD__ || __OpenBSD__

#if (TBB_USE_GCC_BUILTINS && __TBB_GCC_BUILTIN_ATOMICS_PRESENT)
#include "machine/gcc_generic.h"
#elif (TBB_USE_ICC_BUILTINS && __TBB_ICC_BUILTIN_ATOMICS_PRESENT)
#include "machine/icc_generic.h"
#elif __i386__
#include "machine/linux_ia32.h"
#elif __x86_64__
#include "machine/linux_intel64.h"
#elif __ia64__
#include "machine/linux_ia64.h"
#elif __powerpc__
#include "machine/mac_ppc.h"
#elif __ARM_ARCH_7A__ || __aarch64__
#include "machine/gcc_arm.h"
#elif __TBB_GCC_BUILTIN_ATOMICS_PRESENT
#include "machine/gcc_generic.h"
#endif
#include "machine/linux_common.h"

#elif __APPLE__
#if (TBB_USE_ICC_BUILTINS && __TBB_ICC_BUILTIN_ATOMICS_PRESENT)
#include "machine/icc_generic.h"
#elif __TBB_x86_32
#include "machine/linux_ia32.h"
#elif __TBB_x86_64
#include "machine/linux_intel64.h"
#elif __POWERPC__
#include "machine/mac_ppc.h"
#endif
#include "machine/macos_common.h"

#elif _AIX

#include "machine/ibm_aix51.h"

#elif __sun || __SUNPRO_CC

#define __asm__ asm
#define __volatile__ volatile

#if __i386  || __i386__
#include "machine/linux_ia32.h"
#elif __x86_64__
#include "machine/linux_intel64.h"
#elif __sparc
#include "machine/sunos_sparc.h"
#endif
#include <sched.h>

#define __TBB_Yield() sched_yield()

#endif 

#ifndef __TBB_64BIT_ATOMICS
#define __TBB_64BIT_ATOMICS 1
#endif

#if __TBB_USE_FENCED_ATOMICS
#define __TBB_machine_cmpswp1   __TBB_machine_cmpswp1full_fence
#define __TBB_machine_cmpswp2   __TBB_machine_cmpswp2full_fence
#define __TBB_machine_cmpswp4   __TBB_machine_cmpswp4full_fence
#define __TBB_machine_cmpswp8   __TBB_machine_cmpswp8full_fence

#if __TBB_WORDSIZE==8
#define __TBB_machine_fetchadd8             __TBB_machine_fetchadd8full_fence
#define __TBB_machine_fetchstore8           __TBB_machine_fetchstore8full_fence
#define __TBB_FetchAndAddWrelease(P,V)      __TBB_machine_fetchadd8release(P,V)
#define __TBB_FetchAndIncrementWacquire(P)  __TBB_machine_fetchadd8acquire(P,1)
#define __TBB_FetchAndDecrementWrelease(P)  __TBB_machine_fetchadd8release(P,(-1))
#else
#define __TBB_machine_fetchadd4             __TBB_machine_fetchadd4full_fence
#define __TBB_machine_fetchstore4           __TBB_machine_fetchstore4full_fence
#define __TBB_FetchAndAddWrelease(P,V)      __TBB_machine_fetchadd4release(P,V)
#define __TBB_FetchAndIncrementWacquire(P)  __TBB_machine_fetchadd4acquire(P,1)
#define __TBB_FetchAndDecrementWrelease(P)  __TBB_machine_fetchadd4release(P,(-1))
#endif 
#else 
#define __TBB_FetchAndAddWrelease(P,V)      __TBB_FetchAndAddW(P,V)
#define __TBB_FetchAndIncrementWacquire(P)  __TBB_FetchAndAddW(P,1)
#define __TBB_FetchAndDecrementWrelease(P)  __TBB_FetchAndAddW(P,(-1))
#endif 

#if __TBB_WORDSIZE==4
#define __TBB_CompareAndSwapW(P,V,C)    __TBB_machine_cmpswp4(P,V,C)
#define __TBB_FetchAndAddW(P,V)         __TBB_machine_fetchadd4(P,V)
#define __TBB_FetchAndStoreW(P,V)       __TBB_machine_fetchstore4(P,V)
#elif  __TBB_WORDSIZE==8
#if __TBB_USE_GENERIC_DWORD_LOAD_STORE || __TBB_USE_GENERIC_DWORD_FETCH_ADD || __TBB_USE_GENERIC_DWORD_FETCH_STORE
#error These macros should only be used on 32-bit platforms.
#endif

#define __TBB_CompareAndSwapW(P,V,C)    __TBB_machine_cmpswp8(P,V,C)
#define __TBB_FetchAndAddW(P,V)         __TBB_machine_fetchadd8(P,V)
#define __TBB_FetchAndStoreW(P,V)       __TBB_machine_fetchstore8(P,V)
#else 
#error Unsupported machine word size.
#endif 

#ifndef __TBB_Pause
inline void __TBB_Pause(int32_t) {
__TBB_Yield();
}
#endif

namespace tbb {

inline void atomic_fence () { __TBB_full_memory_fence(); }

namespace internal { 


class atomic_backoff : no_copy {

static const int32_t LOOPS_BEFORE_YIELD = 16;
int32_t count;
public:
atomic_backoff() : count(1) {}
atomic_backoff( bool ) : count(1) { pause(); }

void pause() {
if( count<=LOOPS_BEFORE_YIELD ) {
__TBB_Pause(count);
count*=2;
} else {
__TBB_Yield();
}
}

bool bounded_pause() {
__TBB_Pause(count);
if( count<LOOPS_BEFORE_YIELD ) {
count*=2;
return true;
} else {
return false;
}
}

void reset() {
count = 1;
}
};


template<typename T, typename U>
void spin_wait_while_eq( const volatile T& location, U value ) {
atomic_backoff backoff;
while( location==value ) backoff.pause();
}


template<typename T, typename U>
void spin_wait_until_eq( const volatile T& location, const U value ) {
atomic_backoff backoff;
while( location!=value ) backoff.pause();
}

template <typename predicate_type>
void spin_wait_while(predicate_type condition){
atomic_backoff backoff;
while( condition() ) backoff.pause();
}

#ifndef __TBB_ENDIANNESS
#define __TBB_ENDIANNESS __TBB_ENDIAN_DETECT
#endif

#if __TBB_USE_GENERIC_PART_WORD_CAS && __TBB_ENDIANNESS==__TBB_ENDIAN_UNSUPPORTED
#error Generic implementation of part-word CAS may not be used with __TBB_ENDIAN_UNSUPPORTED
#endif

#if __TBB_ENDIANNESS!=__TBB_ENDIAN_UNSUPPORTED
template<typename T>
inline T __TBB_MaskedCompareAndSwap (volatile T * const ptr, const T value, const T comparand ) {
struct endianness{ static bool is_big_endian(){
#if __TBB_ENDIANNESS==__TBB_ENDIAN_DETECT
const uint32_t probe = 0x03020100;
return (((const char*)(&probe))[0]==0x03);
#elif __TBB_ENDIANNESS==__TBB_ENDIAN_BIG || __TBB_ENDIANNESS==__TBB_ENDIAN_LITTLE
return __TBB_ENDIANNESS==__TBB_ENDIAN_BIG;
#else
#error Unexpected value of __TBB_ENDIANNESS
#endif
}};

const uint32_t byte_offset            = (uint32_t) ((uintptr_t)ptr & 0x3);
volatile uint32_t * const aligned_ptr = (uint32_t*)((uintptr_t)ptr - byte_offset );

const uint32_t bits_to_shift     = 8*(endianness::is_big_endian() ? (4 - sizeof(T) - (byte_offset)) : byte_offset);
const uint32_t mask              = (((uint32_t)1<<(sizeof(T)*8)) - 1 )<<bits_to_shift;
const uint32_t shifted_comparand = ((uint32_t)comparand << bits_to_shift)&mask;
const uint32_t shifted_value     = ((uint32_t)value     << bits_to_shift)&mask;

for( atomic_backoff b;;b.pause() ) {
const uint32_t surroundings  = *aligned_ptr & ~mask ; 
const uint32_t big_comparand = surroundings | shifted_comparand ;
const uint32_t big_value     = surroundings | shifted_value     ;
const uint32_t big_result = (uint32_t)__TBB_machine_cmpswp4( aligned_ptr, big_value, big_comparand );
if( big_result == big_comparand                    
|| ((big_result ^ big_comparand) & mask) != 0)   
{
return T((big_result & mask) >> bits_to_shift);
}
else continue;                                     
}
}
#endif 

template<size_t S, typename T>
inline T __TBB_CompareAndSwapGeneric (volatile void *ptr, T value, T comparand );

template<>
inline int8_t __TBB_CompareAndSwapGeneric <1,int8_t> (volatile void *ptr, int8_t value, int8_t comparand ) {
#if __TBB_USE_GENERIC_PART_WORD_CAS
return __TBB_MaskedCompareAndSwap<int8_t>((volatile int8_t *)ptr,value,comparand);
#else
return __TBB_machine_cmpswp1(ptr,value,comparand);
#endif
}

template<>
inline int16_t __TBB_CompareAndSwapGeneric <2,int16_t> (volatile void *ptr, int16_t value, int16_t comparand ) {
#if __TBB_USE_GENERIC_PART_WORD_CAS
return __TBB_MaskedCompareAndSwap<int16_t>((volatile int16_t *)ptr,value,comparand);
#else
return __TBB_machine_cmpswp2(ptr,value,comparand);
#endif
}

template<>
inline int32_t __TBB_CompareAndSwapGeneric <4,int32_t> (volatile void *ptr, int32_t value, int32_t comparand ) {
return (int32_t)__TBB_machine_cmpswp4(ptr,value,comparand);
}

#if __TBB_64BIT_ATOMICS
template<>
inline int64_t __TBB_CompareAndSwapGeneric <8,int64_t> (volatile void *ptr, int64_t value, int64_t comparand ) {
return __TBB_machine_cmpswp8(ptr,value,comparand);
}
#endif

template<size_t S, typename T>
inline T __TBB_FetchAndAddGeneric (volatile void *ptr, T addend) {
T result;
for( atomic_backoff b;;b.pause() ) {
result = *reinterpret_cast<volatile T *>(ptr);
if( __TBB_CompareAndSwapGeneric<S,T> ( ptr, result+addend, result )==result )
break;
}
return result;
}

template<size_t S, typename T>
inline T __TBB_FetchAndStoreGeneric (volatile void *ptr, T value) {
T result;
for( atomic_backoff b;;b.pause() ) {
result = *reinterpret_cast<volatile T *>(ptr);
if( __TBB_CompareAndSwapGeneric<S,T> ( ptr, value, result )==result )
break;
}
return result;
}

#if __TBB_USE_GENERIC_PART_WORD_CAS
#define __TBB_machine_cmpswp1 tbb::internal::__TBB_CompareAndSwapGeneric<1,int8_t>
#define __TBB_machine_cmpswp2 tbb::internal::__TBB_CompareAndSwapGeneric<2,int16_t>
#endif

#if __TBB_USE_GENERIC_FETCH_ADD || __TBB_USE_GENERIC_PART_WORD_FETCH_ADD
#define __TBB_machine_fetchadd1 tbb::internal::__TBB_FetchAndAddGeneric<1,int8_t>
#define __TBB_machine_fetchadd2 tbb::internal::__TBB_FetchAndAddGeneric<2,int16_t>
#endif

#if __TBB_USE_GENERIC_FETCH_ADD
#define __TBB_machine_fetchadd4 tbb::internal::__TBB_FetchAndAddGeneric<4,int32_t>
#endif

#if __TBB_USE_GENERIC_FETCH_ADD || __TBB_USE_GENERIC_DWORD_FETCH_ADD
#define __TBB_machine_fetchadd8 tbb::internal::__TBB_FetchAndAddGeneric<8,int64_t>
#endif

#if __TBB_USE_GENERIC_FETCH_STORE || __TBB_USE_GENERIC_PART_WORD_FETCH_STORE
#define __TBB_machine_fetchstore1 tbb::internal::__TBB_FetchAndStoreGeneric<1,int8_t>
#define __TBB_machine_fetchstore2 tbb::internal::__TBB_FetchAndStoreGeneric<2,int16_t>
#endif

#if __TBB_USE_GENERIC_FETCH_STORE
#define __TBB_machine_fetchstore4 tbb::internal::__TBB_FetchAndStoreGeneric<4,int32_t>
#endif

#if __TBB_USE_GENERIC_FETCH_STORE || __TBB_USE_GENERIC_DWORD_FETCH_STORE
#define __TBB_machine_fetchstore8 tbb::internal::__TBB_FetchAndStoreGeneric<8,int64_t>
#endif

#if __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE
#define __TBB_MACHINE_DEFINE_ATOMIC_SELECTOR_FETCH_STORE(S)                                             \
atomic_selector<S>::word atomic_selector<S>::fetch_store ( volatile void* location, word value ) {  \
return __TBB_machine_fetchstore##S( location, value );                                          \
}

__TBB_MACHINE_DEFINE_ATOMIC_SELECTOR_FETCH_STORE(1)
__TBB_MACHINE_DEFINE_ATOMIC_SELECTOR_FETCH_STORE(2)
__TBB_MACHINE_DEFINE_ATOMIC_SELECTOR_FETCH_STORE(4)
__TBB_MACHINE_DEFINE_ATOMIC_SELECTOR_FETCH_STORE(8)

#undef __TBB_MACHINE_DEFINE_ATOMIC_SELECTOR_FETCH_STORE
#endif 

#if __TBB_USE_GENERIC_DWORD_LOAD_STORE

#if ! __TBB_USE_FENCED_ATOMICS

#define   __TBB_machine_cmpswp8full_fence __TBB_machine_cmpswp8
#endif
__TBB_MACHINE_DEFINE_LOAD8_GENERIC_FENCED(full_fence)
__TBB_MACHINE_DEFINE_STORE8_GENERIC_FENCED(full_fence)

#if ! __TBB_USE_FENCED_ATOMICS
#undef   __TBB_machine_cmpswp8full_fence
#endif

#define __TBB_machine_store8 tbb::internal::__TBB_machine_generic_store8full_fence
#define __TBB_machine_load8  tbb::internal::__TBB_machine_generic_load8full_fence
#endif 

#if __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE

template <typename T, size_t S>
struct machine_load_store {
static T load_with_acquire ( const volatile T& location ) {
T to_return = location;
__TBB_acquire_consistency_helper();
return to_return;
}
static void store_with_release ( volatile T &location, T value ) {
__TBB_release_consistency_helper();
location = value;
}
};

#if __TBB_WORDSIZE==4 && __TBB_64BIT_ATOMICS
template <typename T>
struct machine_load_store<T,8> {
static T load_with_acquire ( const volatile T& location ) {
return (T)__TBB_machine_load8( (const volatile void*)&location );
}
static void store_with_release ( volatile T& location, T value ) {
__TBB_machine_store8( (volatile void*)&location, (int64_t)value );
}
};
#endif 
#endif 

#if __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE
template <typename T, size_t S>
struct machine_load_store_seq_cst {
static T load ( const volatile T& location ) {
__TBB_full_memory_fence();
return machine_load_store<T,S>::load_with_acquire( location );
}
#if __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE
static void store ( volatile T &location, T value ) {
atomic_selector<S>::fetch_store( (volatile void*)&location, (typename atomic_selector<S>::word)value );
}
#else 
static void store ( volatile T &location, T value ) {
machine_load_store<T,S>::store_with_release( location, value );
__TBB_full_memory_fence();
}
#endif 
};

#if __TBB_WORDSIZE==4 && __TBB_64BIT_ATOMICS

template <typename T>
struct machine_load_store_seq_cst<T,8> {
static T load ( const volatile T& location ) {
const int64_t anyvalue = 2305843009213693951LL;
return __TBB_machine_cmpswp8( (volatile void*)const_cast<volatile T*>(&location), anyvalue, anyvalue );
}
static void store ( volatile T &location, T value ) {
#if __TBB_GCC_VERSION >= 40702
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
int64_t result = (volatile int64_t&)location;
#if __TBB_GCC_VERSION >= 40702
#pragma GCC diagnostic pop
#endif
while ( __TBB_machine_cmpswp8((volatile void*)&location, (int64_t)value, result) != result )
result = (volatile int64_t&)location;
}
};
#endif 
#endif 

#if __TBB_USE_GENERIC_RELAXED_LOAD_STORE

template <typename T, size_t S>
struct machine_load_store_relaxed {
static inline T load ( const volatile T& location ) {
return location;
}
static inline void store ( volatile T& location, T value ) {
location = value;
}
};

#if __TBB_WORDSIZE==4 && __TBB_64BIT_ATOMICS
template <typename T>
struct machine_load_store_relaxed<T,8> {
static inline T load ( const volatile T& location ) {
return (T)__TBB_machine_load8( (const volatile void*)&location );
}
static inline void store ( volatile T& location, T value ) {
__TBB_machine_store8( (volatile void*)&location, (int64_t)value );
}
};
#endif 
#endif 

#undef __TBB_WORDSIZE 

template<typename T>
inline T __TBB_load_with_acquire(const volatile T &location) {
return machine_load_store<T,sizeof(T)>::load_with_acquire( location );
}
template<typename T, typename V>
inline void __TBB_store_with_release(volatile T& location, V value) {
machine_load_store<T,sizeof(T)>::store_with_release( location, T(value) );
}
inline void __TBB_store_with_release(volatile size_t& location, size_t value) {
machine_load_store<size_t,sizeof(size_t)>::store_with_release( location, value );
}

template<typename T>
inline T __TBB_load_full_fence(const volatile T &location) {
return machine_load_store_seq_cst<T,sizeof(T)>::load( location );
}
template<typename T, typename V>
inline void __TBB_store_full_fence(volatile T& location, V value) {
machine_load_store_seq_cst<T,sizeof(T)>::store( location, T(value) );
}
inline void __TBB_store_full_fence(volatile size_t& location, size_t value) {
machine_load_store_seq_cst<size_t,sizeof(size_t)>::store( location, value );
}

template<typename T>
inline T __TBB_load_relaxed (const volatile T& location) {
return machine_load_store_relaxed<T,sizeof(T)>::load( const_cast<T&>(location) );
}
template<typename T, typename V>
inline void __TBB_store_relaxed ( volatile T& location, V value ) {
machine_load_store_relaxed<T,sizeof(T)>::store( const_cast<T&>(location), T(value) );
}
inline void __TBB_store_relaxed ( volatile size_t& location, size_t value ) {
machine_load_store_relaxed<size_t,sizeof(size_t)>::store( const_cast<size_t&>(location), value );
}

#ifndef __TBB_TypeWithAlignmentAtLeastAsStrict

#if __TBB_ALIGNAS_PRESENT

#define __TBB_DefineTypeWithAlignment(PowerOf2)       \
struct alignas(PowerOf2) __TBB_machine_type_with_alignment_##PowerOf2 { \
uint32_t member[PowerOf2/sizeof(uint32_t)];       \
};
#define __TBB_alignof(T) alignof(T)

#elif __TBB_ATTRIBUTE_ALIGNED_PRESENT

#define __TBB_DefineTypeWithAlignment(PowerOf2)       \
struct __TBB_machine_type_with_alignment_##PowerOf2 { \
uint32_t member[PowerOf2/sizeof(uint32_t)];       \
} __attribute__((aligned(PowerOf2)));
#define __TBB_alignof(T) __alignof__(T)

#elif __TBB_DECLSPEC_ALIGN_PRESENT

#define __TBB_DefineTypeWithAlignment(PowerOf2)       \
__declspec(align(PowerOf2))                           \
struct __TBB_machine_type_with_alignment_##PowerOf2 { \
uint32_t member[PowerOf2/sizeof(uint32_t)];       \
};
#define __TBB_alignof(T) __alignof(T)

#else 
#error Must define __TBB_TypeWithAlignmentAtLeastAsStrict(T)
#endif


__TBB_DefineTypeWithAlignment(8) 
__TBB_DefineTypeWithAlignment(16)
__TBB_DefineTypeWithAlignment(32)
__TBB_DefineTypeWithAlignment(64)

typedef __TBB_machine_type_with_alignment_64 __TBB_machine_type_with_strictest_alignment;

template<size_t N> struct type_with_alignment;

template<> struct type_with_alignment<1> { char member; };
template<> struct type_with_alignment<2> { uint16_t member; };
template<> struct type_with_alignment<4> { uint32_t member; };
template<> struct type_with_alignment<8> { __TBB_machine_type_with_alignment_8 member; };
template<> struct type_with_alignment<16> {__TBB_machine_type_with_alignment_16 member; };
template<> struct type_with_alignment<32> {__TBB_machine_type_with_alignment_32 member; };
template<> struct type_with_alignment<64> {__TBB_machine_type_with_alignment_64 member; };

#if __TBB_ALIGNOF_NOT_INSTANTIATED_TYPES_BROKEN

template<size_t Size, typename T>
struct work_around_alignment_bug {
static const size_t alignment = __TBB_alignof(T);
};
#define __TBB_TypeWithAlignmentAtLeastAsStrict(T) tbb::internal::type_with_alignment<tbb::internal::work_around_alignment_bug<sizeof(T),T>::alignment>
#else
#define __TBB_TypeWithAlignmentAtLeastAsStrict(T) tbb::internal::type_with_alignment<__TBB_alignof(T)>
#endif  

#endif  

template<typename T>
struct reverse {
static const T byte_table[256];
};
template<typename T>
const T reverse<T>::byte_table[256] = {
0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0,
0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8,
0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4,
0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC,
0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2,
0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6,
0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9,
0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7,
0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

} 
} 

using tbb::internal::__TBB_load_with_acquire;
using tbb::internal::__TBB_store_with_release;

#define __TBB_load_acquire  __TBB_load_with_acquire
#define __TBB_store_release __TBB_store_with_release

#ifndef __TBB_Log2
inline intptr_t __TBB_Log2( uintptr_t x ) {
if( x==0 ) return -1;
intptr_t result = 0;

#if !defined(_M_ARM)
uintptr_t tmp_;
if( sizeof(x)>4 && (tmp_ = ((uint64_t)x)>>32) ) { x=tmp_; result += 32; }
#endif
if( uintptr_t tmp = x>>16 ) { x=tmp; result += 16; }
if( uintptr_t tmp = x>>8 )  { x=tmp; result += 8; }
if( uintptr_t tmp = x>>4 )  { x=tmp; result += 4; }
if( uintptr_t tmp = x>>2 )  { x=tmp; result += 2; }

return (x&2)? result+1: result;
}
#endif

#ifndef __TBB_AtomicOR
inline void __TBB_AtomicOR( volatile void *operand, uintptr_t addend ) {
for( tbb::internal::atomic_backoff b;;b.pause() ) {
uintptr_t tmp = *(volatile uintptr_t *)operand;
uintptr_t result = __TBB_CompareAndSwapW(operand, tmp|addend, tmp);
if( result==tmp ) break;
}
}
#endif

#ifndef __TBB_AtomicAND
inline void __TBB_AtomicAND( volatile void *operand, uintptr_t addend ) {
for( tbb::internal::atomic_backoff b;;b.pause() ) {
uintptr_t tmp = *(volatile uintptr_t *)operand;
uintptr_t result = __TBB_CompareAndSwapW(operand, tmp&addend, tmp);
if( result==tmp ) break;
}
}
#endif

#if __TBB_PREFETCHING
#ifndef __TBB_cl_prefetch
#error This platform does not define cache management primitives required for __TBB_PREFETCHING
#endif

#ifndef __TBB_cl_evict
#define __TBB_cl_evict(p)
#endif
#endif

#ifndef __TBB_Flag
typedef unsigned char __TBB_Flag;
#endif
typedef __TBB_atomic __TBB_Flag __TBB_atomic_flag;

#ifndef __TBB_TryLockByte
inline bool __TBB_TryLockByte( __TBB_atomic_flag &flag ) {
return __TBB_machine_cmpswp1(&flag,1,0)==0;
}
#endif

#ifndef __TBB_LockByte
inline __TBB_Flag __TBB_LockByte( __TBB_atomic_flag& flag ) {
tbb::internal::atomic_backoff backoff;
while( !__TBB_TryLockByte(flag) ) backoff.pause();
return 0;
}
#endif

#ifndef  __TBB_UnlockByte
#define __TBB_UnlockByte(addr) __TBB_store_with_release((addr),0)
#endif

#if ( __TBB_x86_32 || __TBB_x86_64 )  
inline void __TBB_TryLockByteElidedCancel() { __TBB_machine_try_lock_elided_cancel(); }

inline bool __TBB_TryLockByteElided( __TBB_atomic_flag& flag ) {
bool res = __TBB_machine_try_lock_elided( &flag )!=0;
if( !res ) __TBB_TryLockByteElidedCancel();
return res;
}

inline void __TBB_LockByteElided( __TBB_atomic_flag& flag )
{
for(;;) {
tbb::internal::spin_wait_while_eq( flag, 1 );
if( __TBB_machine_try_lock_elided( &flag ) )
return;
__TBB_TryLockByteElidedCancel();
}
}

inline void __TBB_UnlockByteElided( __TBB_atomic_flag& flag ) {
__TBB_machine_unlock_elided( &flag );
}
#endif

#ifndef __TBB_ReverseByte
inline unsigned char __TBB_ReverseByte(unsigned char src) {
return tbb::internal::reverse<unsigned char>::byte_table[src];
}
#endif

template<typename T>
T __TBB_ReverseBits(T src) {
T dst;
unsigned char *original = (unsigned char *) &src;
unsigned char *reversed = (unsigned char *) &dst;

for( int i = sizeof(T)-1; i >= 0; i-- )
reversed[i] = __TBB_ReverseByte( original[sizeof(T)-i-1] );

return dst;
}

#endif 
