#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#endif
int main()
{
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
#pragma omp parallel 
{
printf("The parallel region is executed by thread %d\n",
omp_get_thread_num());
if ( omp_get_thread_num() == 2 ) {
printf("  Thread %d does things differently\n",
omp_get_thread_num());
}
}  
return(0);
}
