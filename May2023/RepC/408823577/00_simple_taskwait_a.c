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
int me = omp_get_thread_num();
#pragma omp single nowait
{      
printf( " »Yuk yuk, here is thread %d from "
"within the single region\n", me );
#pragma omp task
{
printf( "\tHi, here is thread %d "
"running task A\n", me );
}
#pragma omp task
{
printf( "\tHi, here is thread %d "
"running task B\n", me );
}
#pragma omp taskwait
printf(" «Yuk yuk, it is still me, thread %d "
"inside single region after all tasks ended\n", me );
}
printf(" :Hi, here is thread %d after the end "
"of the single region, I'm not waiting "
"all the others\n", me );    
#pragma omp barrier
printf(" +Hi there, finally that's me, thread %d "
"at the end of the parallel region after all tasks ended\n",
omp_get_thread_num());
}
return 0;
}
