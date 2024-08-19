#include <omp.h>
#include <stdio.h>
int main ()  
{
#pragma omp parallel
{
int id = omp_get_thread_num();
printf("Hello World from thread = %d of %d threads.\n", id, omp_get_num_threads());
}
return 0;
}