#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
int main( int argc, char **argv )
{
#pragma omp parallel
{
#pragma omp single
{
printf( " »Yuk yuk, here is thread %d from "
"within the single region\n", omp_get_thread_num() );
#pragma omp task
{
printf( "\tHi, here is thread %d "
"running task A\n", omp_get_thread_num() );
}
#pragma omp task
{
printf( "\tHi, here is thread %d "
"running task B\n", omp_get_thread_num() );
}
}
printf(" :Hi, here is thread %d after the end "
"of the single region, I was stuck waiting "
"all the others\n", omp_get_thread_num() );    
}
return 0;
}
