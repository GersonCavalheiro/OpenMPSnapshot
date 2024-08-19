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
#define SUM_INIT 0
int main()
{
int i, n = 25;
int sum, a[n];
int ref = SUM_INIT + (n-1)*n/2;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
#endif
for (i=0; i<n; i++)
a[i] = i;
#pragma omp parallel
{
#pragma omp single
printf("Number of threads is %d\n",omp_get_num_threads());
}
sum = SUM_INIT;
printf("Value of sum prior to parallel region: %d\n",sum);
#pragma omp parallel for default(none) shared(n,a) reduction(+:sum)
for (i=0; i<n; i++)
sum += a[i];
printf("Value of sum after parallel region: %d\n",sum);
printf("Check results: sum = %d (should be %d)\n",sum,ref);
return(0);
}
