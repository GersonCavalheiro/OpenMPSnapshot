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
int n = 9;
int i, a, b[n];
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
for (i=0; i<n; i++)
b[i] = -1;
#pragma omp parallel shared(a,b) private(i)
{
#pragma omp master
{
a = 10;
printf("Master construct is executed by thread %d\n",
omp_get_thread_num());
}
#pragma omp barrier
#pragma omp for
for (i=0; i<n; i++)
b[i] = a;
} 
printf("After the parallel region:\n");
for (i=0; i<n; i++)
printf("b[%d] = %d\n",i,b[i]);
return(0);
}
