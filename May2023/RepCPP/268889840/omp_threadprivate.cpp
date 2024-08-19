#include <iostream>
#include <omp.h>
#include <unistd.h> 
#include <vector>

int Number = 5;
#pragma omp threadprivate(Number)

int main()
{
int n_threads, thread_id;

std::cout << ">> Initial value of Number = " << Number << std::endl;

#pragma omp parallel private(n_threads, thread_id)
{
thread_id = omp_get_thread_num();
#pragma omp master
{
n_threads = omp_get_num_threads();
std::cout << "Parallel Region Start ... (thread #" << thread_id << ")" << std::endl;
std::cout << "Number of threads = " << n_threads << std::endl;
}
#pragma omp barrier 
usleep(5000 * thread_id);
Number += thread_id + 1;
std::cout << "Number = " << Number << " (at Thread #" << thread_id << ")." << std::endl;
}
std::cout << "Parallel Region End ... (thread #" << thread_id << ")" << std::endl;
std::cout << ">> Final value of Number = " << Number << std::endl;
}
