
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char**argv) {

#pragma omp parallel
{
int rank = omp_get_thread_num();
int nb_threads = omp_get_num_threads();
printf("Hello from thread %d/%d\n", rank, nb_threads);
}

return EXIT_SUCCESS;
}
