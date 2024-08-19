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
(void) omp_set_num_threads(4);
#endif
for (int n=5; n<11; n+=5)
{ 
#pragma omp parallel if (n > 5) num_threads(n) default(none) private(TID) shared(n)
{
TID = omp_get_thread_num();
#pragma omp single
{
printf("Value of n = %d\n",n);
printf("Number of threads in parallel region: %d\n",
omp_get_num_threads());
}
printf("Print statement executed by thread %d\n",TID);
}  
}
return(0);
}
