#include "iostream"
#include "omp.h"

void thread_info(int num) {
if (omp_in_parallel()) {
int thread_id = omp_get_thread_num();
int thread_num = omp_get_num_threads();
printf("[%i] Threads numbers %d || Thread id is %d\n",
num, thread_num, thread_id);
}
}

int main() {
int val = 3;
#pragma omp parallel if (val > 1) num_threads(val)
{
thread_info(1);
}
val = 1;
#pragma omp parallel if (val > 1) num_threads(val)
{
thread_info(2);
}
}

