#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_num_threads() 1
#endif
int main()
{
int ic, i, n = 7;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
#endif
ic = 0;
#pragma omp parallel for shared(ic,n) private(i)
for (i=0; i<n; i++)
{
#pragma omp atomic
ic += 1;
}
printf("Counter = %d\n",ic);
return(0);
}
