#include <omp.h>
extern void abort ();
static int errors = 0;
static int thrs = 4;
int
main ()
{
omp_set_dynamic (0);
#pragma omp parallel num_threads (thrs)
{
static int shrd = 0;
#pragma omp atomic
shrd += 1;
#pragma omp barrier
if (shrd != thrs)
#pragma omp atomic
errors += 1;
}
if (errors)
abort ();
return 0;
}
