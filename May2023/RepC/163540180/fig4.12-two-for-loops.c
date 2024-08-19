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
int i, n = 9;
int a[n], b[n];
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
#pragma omp parallel default(none) shared(n,a,b) private(i)
{
#pragma omp single
printf("First for-loop: number of threads is %d\n",
omp_get_num_threads());
#pragma omp for schedule(runtime)
for (i=0; i<n; i++)
{
printf("Thread %d executes loop iteration %d\n",
omp_get_thread_num(),i);
a[i] = i;
}
#pragma omp single
printf("Second for-loop: number of threads is %d\n",
omp_get_num_threads());
#pragma omp for schedule(runtime)
for (i=0; i<n; i++)
{
printf("Thread %d executes loop iteration %d\n",
omp_get_thread_num(),i);
b[i] = 2 * a[i];
}
} 
return(0);
}
