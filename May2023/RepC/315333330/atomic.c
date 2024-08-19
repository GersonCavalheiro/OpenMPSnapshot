#include "libgomp.h"
static gomp_mutex_t atomic_lock;
void
GOMP_atomic_start (void)
{
gomp_mutex_lock (&atomic_lock);
}
void
GOMP_atomic_end (void)
{
gomp_mutex_unlock (&atomic_lock);
}
#if !GOMP_MUTEX_INIT_0
static void __attribute__((constructor))
initialize_atomic (void)
{
gomp_mutex_init (&atomic_lock);
}
#endif
