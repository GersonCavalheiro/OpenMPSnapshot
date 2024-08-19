#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
int main( int argc, char **argv )
{
int nthreads;
#if defined(_OPENMP)
int order = 0;
#pragma omp parallel              
{   
int my_thread_id = omp_get_thread_num();
#pragma omp master
nthreads = omp_get_num_threads();
#pragma omp critical                   
if ( order == my_thread_id )
{
printf( "\tgreetings from thread num %d\n", my_thread_id );	
order++;		   
}
}
#else
nthreads = 1;
#endif
printf(" %d thread%s greeted you from the %sparallel region\n", nthreads, (nthreads==1)?" has":"s have", (nthreads==1)?"(non)":"" );
return 0;
}
