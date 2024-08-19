#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
double golden_value = 0;
#pragma omp threadprivate( golden_value )
int main( int argc, char **argv )
{
srand48(time(NULL));
int N = 10;  
#pragma omp parallel copyin(golden_value)
{
#pragma omp master
golden_value = 1.618033988;       
#pragma omp barrier
printf("[PR 1] thread %d has a golden value %g\n",
omp_get_thread_num(), golden_value );
}    
#pragma omp parallel copyin(golden_value)
printf("[PR 2] thread %d has a golden value %g\n",
omp_get_thread_num(), golden_value );
return 0;
}
