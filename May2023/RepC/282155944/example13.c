#include <stdio.h>
#include <omp.h>
int main(void) {
#pragma omp parallel sections num_threads(2)
{
#pragma omp section
printf("A: Thread %d\n", omp_get_thread_num());
#pragma omp section
printf("B: Thread %d\n", omp_get_thread_num());
}
return 0;
}