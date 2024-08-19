#include "iostream"
#include "omp.h"

int main() {
#pragma omp parallel num_threads(8)
{
int thread_id, threads_num;
thread_id = omp_get_thread_num();
threads_num = omp_get_num_threads();
printf("Thread id: %d from %d threads. Hello world!\n",
thread_id, threads_num);
}
}

