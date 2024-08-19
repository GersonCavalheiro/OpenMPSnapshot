#pragma GCC visibility push(default)
#define _GNU_SOURCE 
#include <unistd.h>
#include <sys/syscall.h>
#pragma GCC visibility pop
static inline void
futex_wait (int *addr, int val)
{
int err = syscall (SYS_futex, addr, gomp_futex_wait, val, NULL);
if (__builtin_expect (err < 0 && errno == ENOSYS, 0))
{
gomp_futex_wait &= ~FUTEX_PRIVATE_FLAG;
gomp_futex_wake &= ~FUTEX_PRIVATE_FLAG;
syscall (SYS_futex, addr, gomp_futex_wait, val, NULL);
}
}
static inline void
futex_wake (int *addr, int count)
{
int err = syscall (SYS_futex, addr, gomp_futex_wake, count);
if (__builtin_expect (err < 0 && errno == ENOSYS, 0))
{
gomp_futex_wait &= ~FUTEX_PRIVATE_FLAG;
gomp_futex_wake &= ~FUTEX_PRIVATE_FLAG;
syscall (SYS_futex, addr, gomp_futex_wake, count);
}
}
static inline void
cpu_relax (void)
{
__asm volatile ("" : : : "memory");
}
