#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_get_nested() 0
#endif
int main()
{
int TID = -1;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
(void) omp_set_nested(TRUE);
if (! omp_get_nested()) {printf("Warning: nested parallelism not set\n");}
#endif
printf("Nested parallelism is %s\n", 
omp_get_nested() ? "supported" : "not supported");
#pragma omp parallel private(TID)
{
TID = omp_get_thread_num();
printf("Thread %d executes the outer parallel region\n",TID);
#pragma omp parallel num_threads(2) firstprivate(TID)
{
printf("TID %d: Thread %d executes inner parallel region\n",
TID,omp_get_thread_num());
}  
}  
return(0);
}
