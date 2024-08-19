

#if !defined(__TBB_machine_H) || defined(__TBB_machine_windows_intel64_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_windows_intel64_H

#define __TBB_WORDSIZE 8
#define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE

#include "msvc_ia32_common.h"

#if !__TBB_MSVC_PART_WORD_INTERLOCKED_INTRINSICS_PRESENT

#include <intrin.h>
#pragma intrinsic(_InterlockedCompareExchange,_InterlockedExchangeAdd,_InterlockedExchange)
#pragma intrinsic(_InterlockedCompareExchange64,_InterlockedExchangeAdd64,_InterlockedExchange64)

extern "C" {
__int8 __TBB_EXPORTED_FUNC __TBB_machine_cmpswp1 (volatile void *ptr, __int8 value, __int8 comparand );
__int8 __TBB_EXPORTED_FUNC __TBB_machine_fetchadd1 (volatile void *ptr, __int8 addend );
__int8 __TBB_EXPORTED_FUNC __TBB_machine_fetchstore1 (volatile void *ptr, __int8 value );
__int16 __TBB_EXPORTED_FUNC __TBB_machine_cmpswp2 (volatile void *ptr, __int16 value, __int16 comparand );
__int16 __TBB_EXPORTED_FUNC __TBB_machine_fetchadd2 (volatile void *ptr, __int16 addend );
__int16 __TBB_EXPORTED_FUNC __TBB_machine_fetchstore2 (volatile void *ptr, __int16 value );
}

inline long __TBB_machine_cmpswp4 (volatile void *ptr, __int32 value, __int32 comparand ) {
return _InterlockedCompareExchange( (long*)ptr, value, comparand );
}
inline long __TBB_machine_fetchadd4 (volatile void *ptr, __int32 addend ) {
return _InterlockedExchangeAdd( (long*)ptr, addend );
}
inline long __TBB_machine_fetchstore4 (volatile void *ptr, __int32 value ) {
return _InterlockedExchange( (long*)ptr, value );
}

inline __int64 __TBB_machine_cmpswp8 (volatile void *ptr, __int64 value, __int64 comparand ) {
return _InterlockedCompareExchange64( (__int64*)ptr, value, comparand );
}
inline __int64 __TBB_machine_fetchadd8 (volatile void *ptr, __int64 addend ) {
return _InterlockedExchangeAdd64( (__int64*)ptr, addend );
}
inline __int64 __TBB_machine_fetchstore8 (volatile void *ptr, __int64 value ) {
return _InterlockedExchange64( (__int64*)ptr, value );
}

#endif 

#define __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE           1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1
