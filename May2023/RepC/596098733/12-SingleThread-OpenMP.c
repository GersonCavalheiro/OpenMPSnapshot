#include <stdio.h>
#include <omp.h>
int main() {
#pragma omp parallel
{
#pragma omp single
{
printf("I am thread %d. There are %d threads.\n", omp_get_thread_num(), omp_get_num_threads());
}
printf("I am thread %d. Hello World!\n", omp_get_thread_num());
}
return 0;
}
