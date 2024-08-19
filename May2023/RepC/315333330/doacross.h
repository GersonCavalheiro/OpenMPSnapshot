#ifndef GOMP_DOACROSS_H
#define GOMP_DOACROSS_H 1
#include "libgomp.h"
#include <errno.h>
#include "wait.h"
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility push(hidden)
#endif
static inline void doacross_spin (unsigned long *addr, unsigned long expected,
unsigned long cur)
{
do
{
cpu_relax ();
cur = __atomic_load_n (addr, MEMMODEL_RELAXED);
if (expected < cur)
return;
}
while (1);
}
#ifdef HAVE_ATTRIBUTE_VISIBILITY
#pragma GCC visibility pop
#endif
#endif 
