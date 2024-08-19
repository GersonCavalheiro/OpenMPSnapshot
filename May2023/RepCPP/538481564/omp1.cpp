#include <omp.h>
#include <stdio.h>

int main() {
#pragma omp parallel num_threads(8)
printf("Hello World! from thread num %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
}