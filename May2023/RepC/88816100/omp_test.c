#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[]) {
int th_id, nthreads;
#pragma omp parallel private(th_id, nthreads) num_threads(8)
{
th_id = omp_get_thread_num();
nthreads = omp_get_num_threads();
printf("Hello World from thread %d of %d threads.\n", th_id, nthreads);
}
getchar();
return EXIT_SUCCESS;
}
