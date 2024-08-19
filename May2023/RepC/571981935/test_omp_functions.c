#include <omp.h>
#include <stdio.h>
int main() {
printf("Max threads: %d\n", omp_get_max_threads());
#pragma omp parallel
{
int id = omp_get_thread_num();
printf("[%2d] Number of threads used: %d\n",
id, omp_get_num_threads());
}
printf("*** Now setting the number of threads.\n");
omp_set_num_threads(8);
printf("Max threads: %d\n", omp_get_max_threads());
#pragma omp parallel
{
int id = omp_get_thread_num();
printf("[%2d] Number of threads used: %d\n",
id, omp_get_num_threads());
}
}
