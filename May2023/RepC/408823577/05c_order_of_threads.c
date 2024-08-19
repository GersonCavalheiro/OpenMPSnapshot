#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
void do_something( int who_am_I )
{
#pragma omp ordered
printf( "\tgreetings from thread num %d\n", who_am_I );
}
int main( int argc, char **argv )
{
int nthreads;
#if defined(_OPENMP)
#pragma omp parallel
{   
int my_thread_id = omp_get_thread_num();
#pragma omp master
nthreads = omp_get_num_threads();
#pragma omp barrier                           
#pragma omp for ordered                       
for ( int i = 0; i < nthreads; i++)          
do_something( my_thread_id );
}
#else
nthreads = 1;
#endif
printf(" %d thread%s greeted you from the %sparallel region\n", nthreads, (nthreads==1)?" has":"s have", (nthreads==1)?"(non)":"" );
return 0;
}
