#include <stdio.h>
#include <omp.h>
void
skip (int i)
{
}
void
work (int i)
{
}
int
main ()
{
omp_lock_t lck;
int id;
omp_init_lock (&lck);
#pragma omp parallel shared(lck) private(id)
{
id = omp_get_thread_num ();
omp_set_lock (&lck);
printf ("My thread id is %d.\n", id);
omp_unset_lock (&lck);
while (!omp_test_lock (&lck))
{
skip (id);		
}
work (id);			
omp_unset_lock (&lck);
}
omp_destroy_lock (&lck);
return 0;
}
