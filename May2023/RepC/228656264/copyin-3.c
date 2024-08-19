#include <omp.h>
#include <stdlib.h>
int thr;
#pragma omp threadprivate (thr)
int
test (int l)
{
return l || (thr != omp_get_thread_num () * 2);
}
int
main (void)
{
int l = 0;
omp_set_dynamic (0);
omp_set_num_threads (6);
thr = 8;
#pragma omp parallel copyin (thr)
;
#pragma omp parallel reduction (||:l)
{
l = thr != 8;
thr = omp_get_thread_num () * 2;
#pragma omp barrier
l = test (l);
}
if (l)
abort ();
return 0;
}
