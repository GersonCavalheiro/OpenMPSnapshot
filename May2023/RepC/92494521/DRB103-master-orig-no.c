#include <omp.h>
#include <stdio.h>
int main()
{
int k;
#pragma omp parallel
{
#pragma omp master
{
k = omp_get_num_threads();
printf ("Number of Threads requested = %i\n",k);
}
}
return 0;
}
