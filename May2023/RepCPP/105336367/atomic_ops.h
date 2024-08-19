

#ifndef ATOMIC_OPS_H

#define ATOMIC_OPS_H

#include <assert.h>
#include <stddef.h>

















































































































#define AO_t size_t




#define AO_TS_INITIALIZER (AO_t)AO_TS_CLEAR


#if defined(__GNUC__) || defined(_MSC_VER) || defined(__INTEL_COMPILER) \
|| defined(__DMC__) || defined(__WATCOMC__)
# define AO_INLINE static __inline
#elif defined(__sun)
# define AO_INLINE static inline
#else
# define AO_INLINE static
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
# define AO_compiler_barrier() __asm__ __volatile__("" : : : "memory")
#elif defined(_MSC_VER) || defined(__DMC__) || defined(__BORLANDC__) \
|| defined(__WATCOMC__)
# if defined(_AMD64_) || defined(_M_X64) || _MSC_VER >= 1400
#   if defined(_WIN32_WCE)

#   elif defined(_MSC_VER)
#     include <intrin.h>
#   endif
#   pragma intrinsic(_ReadWriteBarrier)
#   define AO_compiler_barrier() _ReadWriteBarrier()


# else
#   define AO_compiler_barrier() __asm { }


# endif
#elif defined(__INTEL_COMPILER)
# define AO_compiler_barrier() __memory_barrier() 
#elif defined(_HPUX_SOURCE)
# if defined(__ia64)
#   include <machine/sys/inline.h>
#   define AO_compiler_barrier() _Asm_sched_fence()
# else


static volatile int AO_barrier_dummy;
#   define AO_compiler_barrier() AO_barrier_dummy = AO_barrier_dummy
# endif
#else


# define AO_compiler_barrier() asm("")
#endif

#if defined(AO_USE_PTHREAD_DEFS)
# include "atomic_ops/sysdeps/generic_pthread.h"
#endif 

#if defined(__GNUC__) && !defined(AO_USE_PTHREAD_DEFS) \
&& !defined(__INTEL_COMPILER)
# if defined(__i386__)



#   include "./x86.h"
# endif 
# if defined(__x86_64__)
#   if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 2)

#     define AO_USE_SYNC_CAS_BUILTIN
#   endif
#   include "./x86_64.h"
# endif 
# if defined(__ia64__)
#   include "./ia64.h"
#   define AO_GENERALIZE_TWICE
# endif 
# if defined(__hppa__)
#   include "atomic_ops/sysdeps/gcc/hppa.h"
#   define AO_CAN_EMUL_CAS
# endif 
# if defined(__alpha__)
#   include "atomic_ops/sysdeps/gcc/alpha.h"
#   define AO_GENERALIZE_TWICE
# endif 
# if defined(__s390__)
#   include "atomic_ops/sysdeps/gcc/s390.h"
# endif 
# if defined(__sparc__)
#   include "./sparc.h"
#   define AO_CAN_EMUL_CAS
# endif 
# if defined(__m68k__)
#   include "atomic_ops/sysdeps/gcc/m68k.h"
# endif 
# if defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) \
|| defined(__powerpc64__) || defined(__ppc64__)
#   include "./powerpc.h"
# endif 
# if defined(__arm__) && !defined(AO_USE_PTHREAD_DEFS)
#   include "atomic_ops/sysdeps/gcc/arm.h"
#   define AO_CAN_EMUL_CAS
# endif 
# if defined(__cris__) || defined(CRIS)
#   include "atomic_ops/sysdeps/gcc/cris.h"
# endif
# if defined(__mips__)
#   include "atomic_ops/sysdeps/gcc/mips.h"
# endif 
# if defined(__sh__) || defined(SH4)
#   include "atomic_ops/sysdeps/gcc/sh.h"
#   define AO_CAN_EMUL_CAS
# endif 
#endif 

#if defined(__INTEL_COMPILER) && !defined(AO_USE_PTHREAD_DEFS)
# if defined(__ia64__)
#   include "./ia64.h"
#   define AO_GENERALIZE_TWICE
# endif
# if defined(__GNUC__)

#   if defined(__i386__)
#     include "./x86.h"
#   endif 
#   if defined(__x86_64__)
#     if __INTEL_COMPILER > 1110
#       define AO_USE_SYNC_CAS_BUILTIN
#     endif
#     include "./x86_64.h"
#   endif 
# endif
#endif

#if defined(_HPUX_SOURCE) && !defined(__GNUC__) && !defined(AO_USE_PTHREAD_DEFS)
# if defined(__ia64)
#   include "atomic_ops/sysdeps/hpc/ia64.h"
#   define AO_GENERALIZE_TWICE
# else
#   include "atomic_ops/sysdeps/hpc/hppa.h"
#   define AO_CAN_EMUL_CAS
# endif
#endif

#if defined(__sun) && !defined(__GNUC__) && !defined(AO_USE_PTHREAD_DEFS)

# if defined(__i386)
#   include "atomic_ops/sysdeps/sunc/x86.h"
# endif 
# if defined(__x86_64) || defined(__amd64)
#   include "atomic_ops/sysdeps/sunc/x86_64.h"
# endif 
#endif

#if !defined(__GNUC__) && (defined(sparc) || defined(__sparc)) \
&& !defined(AO_USE_PTHREAD_DEFS)
#   include "atomic_ops/sysdeps/sunc/sparc.h"
#   define AO_CAN_EMUL_CAS
#endif

#if defined(_MSC_VER) || defined(__DMC__) || defined(__BORLANDC__) \
|| (defined(__WATCOMC__) && defined(__NT__))
# if defined(_AMD64_) || defined(_M_X64)
#   include "atomic_ops/sysdeps/msftc/x86_64.h"
# elif defined(_M_IX86) || defined(x86)
#   include "atomic_ops/sysdeps/msftc/x86.h"
# elif defined(_M_ARM) || defined(ARM) || defined(_ARM_)
#   include "atomic_ops/sysdeps/msftc/arm.h"
# endif
#endif

#if defined(AO_REQUIRE_CAS) && !defined(AO_HAVE_compare_and_swap) \
&& !defined(AO_HAVE_compare_and_swap_full) \
&& !defined(AO_HAVE_compare_and_swap_acquire)
# if defined(AO_CAN_EMUL_CAS)
#   include "atomic_ops/sysdeps/emul_cas.h"
# else
#  error Cannot implement AO_compare_and_swap_full on this architecture.
# endif
#endif  



#if AO_AO_TS_T && !defined(AO_CLEAR)
# define AO_CLEAR(addr) AO_store_release((AO_TS_t *)(addr), AO_TS_CLEAR)
#endif
#if AO_CHAR_TS_T && !defined(AO_CLEAR)
# define AO_CLEAR(addr) AO_char_store_release((AO_TS_t *)(addr), AO_TS_CLEAR)
#endif


#include "./generalize.h"
#ifdef AO_GENERALIZE_TWICE
# include "./generalize.h"
#endif


#define AO_TS_T AO_TS_t
#define AO_T AO_t
#define AO_TS_VAL AO_TS_VAL_t

#endif 
