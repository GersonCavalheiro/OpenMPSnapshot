

#ifndef FF_ASM_GENERIC_ATOMIC_H
#define FF_ASM_GENERIC_ATOMIC_H




#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)

#if defined(__linux__) || defined(__FreeBSD__)
#ifdef __i386__
#include "atomic-i386.h"
#elif __arm__
#include "atomic-arm.h"
#define BITS_PER_LONG 32
#elif __aarch64__
#include "atomic-arm.h"
#define BITS_PER_LONG 64
#elif __x86_64__
#include "atomic-x86_64.h"
#if !defined(BITS_PER_LONG)
#define BITS_PER_LONG 64
#endif
#elif __ia64__
#error "IA64 not yet supported"
#endif

#elif __APPLE__

#ifdef __i386__
#include "atomic-i386.h"
#elif __x86_64__
#include "atomic-x86_64.h"
#if !defined(BITS_PER_LONG)
#define BITS_PER_LONG 64
#endif
#elif __POWERPC__
#include "atomic-ppc.h"
#endif

#endif



#if (BITS_PER_LONG == 64)


typedef atomic64_t atomic_long_t;

#define ATOMIC_LONG_INIT(i) ATOMIC64_INIT(i)

static inline unsigned long atomic_long_read(atomic_long_t *l)
{
atomic64_t *v = (atomic64_t *)l;

return atomic64_read(v);
}

static inline void atomic_long_set(atomic_long_t *l, long i)
{
atomic64_t *v = (atomic64_t *)l;

atomic64_set(v, i);
}

static inline void atomic_long_inc(atomic_long_t *l)
{
atomic64_t *v = (atomic64_t *)l;

atomic64_inc(v);
}

static inline unsigned long atomic_long_inc_return(atomic_long_t *l)
{
atomic64_t *v = (atomic64_t *)l;

return atomic64_inc_return(v);
}

static inline void atomic_long_dec(atomic_long_t *l)
{
atomic64_t *v = (atomic64_t *)l;

atomic64_dec(v);
}

static inline unsigned long atomic_long_dec_return(atomic_long_t *l)
{
atomic64_t *v = (atomic64_t *)l;

return atomic64_dec_return(v);
}

static inline void atomic_long_add(long i, atomic_long_t *l)
{
atomic64_t *v = (atomic64_t *)l;

atomic64_add(i, v);
}

static inline void atomic_long_sub(long i, atomic_long_t *l)
{
atomic64_t *v = (atomic64_t *)l;

atomic64_sub(i, v);
}


static inline unsigned long atomic_long_add_unless(atomic_long_t *l, long a, long u)
{
atomic64_t *v = (atomic64_t *)l;

return atomic64_add_unless(v, a, u);
}


#define atomic_long_cmpxchg(l, old, new) \
(atomic64_cmpxchg((atomic64_t *)(l), (old), (new)))

#else

typedef atomic_t atomic_long_t;

#define ATOMIC_LONG_INIT(i) ATOMIC_INIT(i)

static inline unsigned long atomic_long_read(atomic_long_t *l)
{
atomic_t *v = (atomic_t *)l;

return atomic_read(v);
}

static inline void atomic_long_set(atomic_long_t *l, long i)
{
atomic_t *v = (atomic_t *)l;

atomic_set(v, i);
}

static inline void atomic_long_inc(atomic_long_t *l)
{
atomic_t *v = (atomic_t *)l;

atomic_inc(v);
}

static inline unsigned long atomic_long_inc_return(atomic_long_t *l)
{
atomic_t *v = (atomic_t *)l;

return atomic_inc_return(v);
}

static inline void atomic_long_dec(atomic_long_t *l)
{
atomic_t *v = (atomic_t *)l;

atomic_dec(v);
}

static inline unsigned long atomic_long_dec_return(atomic_long_t *l)
{
atomic_t *v = (atomic_t *)l;

return atomic_dec_return(v);
}


static inline void atomic_long_add(long i, atomic_long_t *l)
{
atomic_t *v = (atomic_t *)l;

atomic_add(i, v);
}

static inline void atomic_long_sub(long i, atomic_long_t *l)
{
atomic_t *v = (atomic_t *)l;

atomic_sub(i, v);
}

static inline unsigned long atomic_long_add_unless(atomic_long_t *l, long a, long u)
{
atomic_t *v = (atomic_t *)l;

return atomic_add_unless(v, a, u);
}

#define atomic_long_cmpxchg(l, old, new) \
(atomic_cmpxchg((atomic_t *)(l), (old), (new)))

#endif

#elif  defined(_WIN32)
typedef __declspec(align(4 )) struct { volatile long counter; } atomic_t;
typedef atomic_t atomic_long_t;
#define atomic_set(v,i)         (((v)->counter) = (i))
#define atomic_read(v)          ((v)->counter)
#define atomic_long_set atomic_set
#define atomic_long_read atomic_read

#define BITS_PER_LONG 32 
#pragma intrinsic (_InterlockedIncrement)
static inline void atomic_long_inc(atomic_long_t *v) {
_InterlockedIncrement(&v->counter);
}
static inline long atomic_long_inc_return(atomic_long_t *v) {

return _InterlockedIncrement(&v->counter);
}

#pragma intrinsic (_InterlockedDecrement)
static inline void atomic_long_dec(atomic_long_t *v) {
_InterlockedDecrement(&v->counter);
}

#endif

#endif 


