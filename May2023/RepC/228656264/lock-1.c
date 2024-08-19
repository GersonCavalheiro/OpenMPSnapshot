#include <omp.h>
#include <stdlib.h>
int
main (void)
{
int l = 0;
omp_nest_lock_t lock;
omp_init_nest_lock (&lock);
if (omp_test_nest_lock (&lock) != 1)
abort ();
if (omp_test_nest_lock (&lock) != 2)
abort ();
#pragma omp parallel if (0) reduction (+:l)
{
if (omp_test_nest_lock (&lock) != 0)
l++;
}
if (l)
abort ();
if (omp_test_nest_lock (&lock) != 3)
abort ();
omp_unset_nest_lock (&lock);
omp_unset_nest_lock (&lock);
omp_unset_nest_lock (&lock);
omp_destroy_nest_lock (&lock);
return 0;
}
