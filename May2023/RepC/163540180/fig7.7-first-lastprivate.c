#include <stdio.h>
#include <stdlib.h>
#define TRUE  1
#define FALSE 0
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
int main ()
{
int a, b, c, i, n;
int a_check, c_check;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
b = 50;
n = 1858;
a_check = b + n-1;
c_check = a_check + b;
printf("Before parallel loop: b = %d n = %d\n",b,n);
#pragma omp parallel for private(i), firstprivate(b), lastprivate(a)
for (i=0; i<n; i++)
{
a = b+i;
} 
c = a + b;
printf("Values of a and c after parallel for:\n");
printf("\ta = %d\t(correct value is %d)\n",a,a_check);
printf("\tc = %d\t(correct value is %d)\n",c,c_check);
return(0);
}
