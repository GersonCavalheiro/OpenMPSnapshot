#include <stdio.h>
#include <omp.h>
int main( int argc, char* argv[] )
{
printf("Before parallel region: threads=%d, max_threads=%d\n",
omp_get_num_threads(), omp_get_max_threads());
#pragma omp parallel
{
printf("Inside parallel region: threads=%d, max_threads=%d\n",
omp_get_num_threads(), omp_get_max_threads());
}
return 0;
}
