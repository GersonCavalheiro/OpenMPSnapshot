#include <omp.h>

int main() {
omp_set_num_threads(16);
int thread = 0;
#pragma omp sections
{
#pragma omp section
thread = omp_get_thread_num();
#pragma omp section
thread = omp_get_thread_num();
#pragma omp section
thread = omp_get_thread_num();
#pragma omp section
thread = omp_get_thread_num();
#pragma omp section
thread = omp_get_thread_num();
#pragma omp section
thread = omp_get_thread_num();
#pragma omp section
thread = omp_get_thread_num();
#pragma omp section
thread = omp_get_thread_num();
}
return thread;
}
