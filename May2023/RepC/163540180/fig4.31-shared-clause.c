#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#endif
int main()
{
int i, n = 7;
int a[n];
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
for (i=0; i<n; i++)
a[i] = i+1;
#pragma omp parallel for shared(a)
for (i=0; i<n; i++)
{
a[i] += i;
} 
printf("In main program after parallel for:\n");
for (i=0; i<n; i++)
printf("a[%d] = %d\n",i,a[i]);
return(0);
}
