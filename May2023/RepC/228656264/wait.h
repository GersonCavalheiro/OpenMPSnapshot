#ifndef GOMP_WAIT_H
#define GOMP_WAIT_H 1
#include "libgomp.h"
#include <errno.h>
#define FUTEX_WAIT	0
#define FUTEX_WAKE	1
#define FUTEX_PRIVATE_FLAG	128
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility push(hidden)
#endif
extern int gomp_futex_wait, gomp_futex_wake;
#include <futex.h>
static inline int do_spin (int *addr, int val)
{
unsigned long long i, count = gomp_spin_count_var;
if (__builtin_expect (__atomic_load_n (&gomp_managed_threads,
MEMMODEL_RELAXED)
> gomp_available_cpus, 0))
count = gomp_throttled_spin_count_var;
for (i = 0; i < count; i++)
if (__builtin_expect (__atomic_load_n (addr, MEMMODEL_RELAXED) != val, 0))
return 0;
else
cpu_relax ();
return 1;
}
static inline void do_wait (int *addr, int val)
{
if (do_spin (addr, val))
futex_wait (addr, val);
}
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility pop
#endif
#endif 
