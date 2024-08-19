#if !defined(_OPENMP)
#error you need to use OpenMP to compile this code, use the appropriated flag for your compiler
#endif
#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
int main( int argc, char **argv )
{
#pragma omp parallel
{
int nthreads = omp_get_num_threads();
#pragma omp single
printf("%d threads in the outer parallel region\n", nthreads);
#pragma omp parallel
{
int nthreads_inner = omp_get_num_threads();
#pragma omp single
printf("\t%d threads in the inner parallel region\n", nthreads_inner);
}
}
return 0;
}
