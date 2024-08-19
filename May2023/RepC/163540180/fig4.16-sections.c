#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#endif
void funcA();
void funcB();
int main()
{
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(4);
#endif
#pragma omp parallel 
{
#pragma omp sections
{
#pragma omp section
(void) funcA();
#pragma omp section
(void) funcB();
} 
} 
return(0);
}
void funcA()
{
printf("In funcA: this section is executed by thread %d\n",
omp_get_thread_num());
}
void funcB()
{
printf("In funcB: this section is executed by thread %d\n",
omp_get_thread_num());
}
