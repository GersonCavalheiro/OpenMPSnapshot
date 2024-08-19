#include <omp.h>
#include <assert.h> 
int main()
{
omp_lock_t lck;
int i=0;
omp_init_lock(&lck);
#pragma omp parallel sections
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
