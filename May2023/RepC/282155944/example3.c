#include <stdio.h>
#include <omp.h>
int main(void) {
#pragma omp parallel num_threads(4)
printf("Thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
return 0;
}