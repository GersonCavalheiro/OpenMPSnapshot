#include <stdio.h>
#include <omp.h>
int main(void) {
int a = 1;
#pragma omp parallel num_threads(4) private(a)
printf("Thread: %d, a = %d\n", omp_get_thread_num(), a++);
return 0;
}