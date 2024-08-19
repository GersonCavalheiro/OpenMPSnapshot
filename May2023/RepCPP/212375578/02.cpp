#include <cstdio>
#include <omp.h>

int main() {
omp_set_num_threads(3);
#pragma omp parallel if (omp_get_max_threads() > 1)
{
printf("1. thread#%d says threads count = %d\n",
omp_get_thread_num(),
omp_get_num_threads());
}

printf("\n");

omp_set_num_threads(1);
#pragma omp parallel if (omp_get_max_threads() > 1)
{
printf("2. thread#%d says threads count = %d\n",
omp_get_thread_num(),
omp_get_num_threads());
}
}
