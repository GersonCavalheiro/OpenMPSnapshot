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
int i, n = 5;
int a, a_shared;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
#endif
#pragma omp parallel for private(i) private(a) shared(a_shared)
for (i=0; i<n; i++)
{
a = i+1;
printf("Thread %d has a value of a = %d for i = %d\n",
omp_get_thread_num(),a,i);
if ( i == n-1 ) a_shared = a;
} 
printf("Value of a after parallel for: a_shared = %d\n",a_shared);
return(0);
}
