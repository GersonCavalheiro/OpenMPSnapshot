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
int sum, TID, a[n];
int ref = SUM_INIT + (n-1)*n/2;
int sumLocal;
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
#pragma omp parallel default(none) shared(n,a,sum) private(TID,sumLocal)
{
TID = omp_get_thread_num();
sumLocal = 0;
#pragma omp for
for (i=0; i<n; i++)
sumLocal += a[i];
#pragma omp critical (update_sum)
{
sum += sumLocal;
printf("TID=%d: sumLocal = %d sum = %d\n",TID,sumLocal,sum);
}
} 
printf("Value of sum after parallel region: %d\n",sum);
printf("Check results: sum = %d (should be %d)\n",sum,ref);
return(0);
}
