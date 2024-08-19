#include "iostream"
#include "omp.h"

int main() {
#pragma omp parallel num_threads(8)
{
int thread_id = omp_get_thread_num();
int threads_num = omp_get_num_threads();
int count = 0;
for (int i = 0; i < 80000000 - 10000000 * (thread_id + 1); i++) {
count++;
}
printf("Thread id: %d from %d threads. Hello world!\n",
thread_id + 1, threads_num);
}
}
