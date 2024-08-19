#include <iostream>
#include <omp.h>

int main()
{
int n_threads, thread_id;

#pragma omp parallel private(n_threads, thread_id)
{
thread_id = omp_get_thread_num();
if (thread_id == 0)
{
n_threads = omp_get_num_threads();
std::cout << "Number of threads: " << n_threads << std::endl;
}
std::cout << "Hello, World! (from thread: #" << thread_id << ")" << std::endl;
}
}