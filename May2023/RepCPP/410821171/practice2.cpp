#include <iostream>
#include <omp.h>

constexpr int NUM = 4;

void print_tid() {
printf("Hello tid: %d\n", omp_get_thread_num());
}

int main() {
#pragma omp parallel
{
#pragma omp single
{
printf("The number of Threads = %d \n\n", omp_get_num_threads());
}
print_tid();
}
printf("\n");
omp_set_num_threads(NUM);

#pragma omp parallel
{
#pragma omp single
{
printf("Get the number of Threads = %d \n\n", omp_get_num_threads());
}
print_tid();
}

return 0;
}