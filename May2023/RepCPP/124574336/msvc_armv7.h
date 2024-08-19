

#if !defined(__TBB_machine_H) || defined(__TBB_msvc_armv7_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_msvc_armv7_H

#include <intrin.h>
#include <float.h>

#define __TBB_WORDSIZE 4

#define __TBB_ENDIANNESS __TBB_ENDIAN_UNSUPPORTED

#if defined(TBB_WIN32_USE_CL_BUILTINS)
#pragma intrinsic(_ReadWriteBarrier)
#pragma intrinsic(_mm_mfence)
#define __TBB_compiler_fence()    _ReadWriteBarrier()
#define __TBB_full_memory_fence() _mm_mfence()
#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#define __TBB_acquire_consistency_helper() __TBB_compiler_fence()
#define __TBB_release_consistency_helper() __TBB_compiler_fence()
#else
#define __TBB_compiler_fence()    __dmb(_ARM_BARRIER_SY)
#define __TBB_full_memory_fence() __dmb(_ARM_BARRIER_SY)
#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#define __TBB_acquire_consistency_helper() __TBB_full_memory_fence()
#define __TBB_release_consistency_helper() __TBB_full_memory_fence()
#endif




#define __TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(S,T,F)                                               \
inline T __TBB_machine_cmpswp##S( volatile void *ptr, T value, T comparand ) {                   \
return _InterlockedCompareExchange##F(reinterpret_cast<volatile T *>(ptr),value,comparand);  \
}                                                                                                \

#define __TBB_MACHINE_DEFINE_ATOMICS_FETCHADD(S,T,F)                                             \
inline T __TBB_machine_fetchadd##S( volatile void *ptr, T value ) {                              \
return _InterlockedExchangeAdd##F(reinterpret_cast<volatile T *>(ptr),value);                \
}                                                                                                \

__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(1,char,8)
__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(2,short,16)
__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(4,long,)
__TBB_MACHINE_DEFINE_ATOMICS_CMPSWP(8,__int64,64)
__TBB_MACHINE_DEFINE_ATOMICS_FETCHADD(4,long,)
#if defined(TBB_WIN32_USE_CL_BUILTINS)
#define __TBB_64BIT_ATOMICS 0
#else
__TBB_MACHINE_DEFINE_ATOMICS_FETCHADD(8,__int64,64)
#endif

inline void __TBB_machine_pause (int32_t delay )
{
while(delay>0)
{
__TBB_compiler_fence();
delay--;
}
}

#define __TBB_CPU_CTL_ENV_PRESENT 1

namespace tbb {
namespace internal {

template <typename T, size_t S>
struct machine_load_store_relaxed {
static inline T load ( const volatile T& location ) {
const T value = location;


__TBB_acquire_consistency_helper();
return value;
}

static inline void store ( volatile T& location, T value ) {
location = value;
}
};

class cpu_ctl_env {
private:
unsigned int my_ctl;
public:
bool operator!=( const cpu_ctl_env& ctl ) const { return my_ctl != ctl.my_ctl; }
void get_env() { my_ctl = _control87(0, 0); }
void set_env() const { _control87( my_ctl, ~0U ); }
};

} 
} 

#define __TBB_CompareAndSwap4(P,V,C) __TBB_machine_cmpswp4(P,V,C)
#define __TBB_CompareAndSwap8(P,V,C) __TBB_machine_cmpswp8(P,V,C)
#define __TBB_Pause(V) __TBB_machine_pause(V)

#define __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE               1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE                1
#define __TBB_USE_GENERIC_PART_WORD_FETCH_ADD                   1
#define __TBB_USE_GENERIC_PART_WORD_FETCH_STORE                 1
#define __TBB_USE_GENERIC_FETCH_STORE                           1
#define __TBB_USE_GENERIC_DWORD_LOAD_STORE                      1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE     1

#if defined(TBB_WIN32_USE_CL_BUILTINS)
#if !__TBB_WIN8UI_SUPPORT
extern "C" __declspec(dllimport) int __stdcall SwitchToThread( void );
#define __TBB_Yield()  SwitchToThread()
#else
#include<thread>
#define __TBB_Yield()  std::this_thread::yield()
#endif
#else
#define __TBB_Yield() __yield()
#endif

#define __TBB_AtomicOR(P,V)     __TBB_machine_OR(P,V)
#define __TBB_AtomicAND(P,V)    __TBB_machine_AND(P,V)

template <typename T1,typename T2>
inline void __TBB_machine_OR( T1 *operand, T2 addend ) {
_InterlockedOr((long volatile *)operand, (long)addend);
}

template <typename T1,typename T2>
inline void __TBB_machine_AND( T1 *operand, T2 addend ) {
_InterlockedAnd((long volatile *)operand, (long)addend);
}

