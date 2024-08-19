#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#endif
int main(int argc, char *argv[])
{
int TID = omp_get_thread_num();
printf("Thread ID of the master thread is %d\n",TID);
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
#pragma omp parallel
{
int TID = omp_get_thread_num();
printf("In parallel region - Thread ID is %d\n",TID);
} 
return(0);
}
