#include <stdio.h>
#include <omp.h>
int main() {
#pragma omp parallel sections num_threads(4)
{
printf("Hello from thread %d\n", omp_get_thread_num());
printf("Hello from thread %d\n", omp_get_thread_num());
}
}
