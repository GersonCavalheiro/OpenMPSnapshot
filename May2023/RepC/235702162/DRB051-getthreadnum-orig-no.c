#include <omp.h>
#include <stdio.h>
int main()
{
int numThreads=0 ; 
#pragma omp parallel
{
if ( omp_get_thread_num()==0 ) {
numThreads = omp_get_num_threads();
}
}
printf ("numThreads=%d\n", numThreads);
return 0;
}
