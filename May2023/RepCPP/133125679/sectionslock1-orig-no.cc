



#include <omp.h>
#include <assert.h>

#if !defined(NTHREADS)
#define NTHREADS 2
#endif

int main()
{
omp_lock_t lck;
int i=0;
omp_init_lock(&lck);
#pragma omp parallel sections num_threads(NTHREADS)
{
#pragma omp section
{
omp_set_lock(&lck);
i += 1;
omp_unset_lock(&lck);
}
#pragma omp section
{
omp_set_lock(&lck);
i += 2;
omp_unset_lock(&lck);
}
}

omp_destroy_lock(&lck);
assert (i==3);
return 0;
}
