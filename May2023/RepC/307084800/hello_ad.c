#include <omp.h>
#include <stdio.h>
int main() {
int i = 0;
#pragma omp parallel shared(i)
{
while (i != omp_get_num_threads()) {
if (i == omp_get_num_threads() - omp_get_thread_num() - 1) {
printf("Hello, world! %d\n", omp_get_thread_num());
#pragma omp atomic 
i++;
}
}
}
}