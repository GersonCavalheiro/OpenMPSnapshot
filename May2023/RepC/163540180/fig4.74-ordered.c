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
int i, TID, n = 9;
int a[n];
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
for (i=0; i<n; i++)
a[i] = i;
#pragma omp parallel for default(none) ordered schedule(runtime) private(i,TID) shared(n,a)
for (i=0; i<n; i++)
{
TID = omp_get_thread_num();
printf("Thread %d updates a[%d]\n",TID,i);
a[i] += i;
#pragma omp ordered
{printf("Thread %d prints value of a[%d] = %d\n",TID,i,a[i]);}
}  
return(0);
}
