#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
int main()
{
int TID;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
#endif
#pragma omp parallel
{
#pragma omp single
printf("Number of threads is %d\n",omp_get_num_threads());
}
#pragma omp parallel default(none) private(TID)
{
TID = omp_get_thread_num();
#pragma omp critical (print_tid)
{
printf("I am thread %d\n",TID);
}
} 
return(0);
}
