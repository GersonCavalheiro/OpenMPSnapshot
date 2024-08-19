#include <iostream>
#include <omp.h>

int main(int argc, char *argv[]) {
omp_set_num_threads(4);
std::cout << "How many threads are there? " << omp_get_num_threads() << std::endl << std::flush;
#pragma omp parallel
#pragma omp master
{
std::cout << "How many threads are there? " << omp_get_num_threads() << std::endl << std::flush;
}
std::cout << "How many threads are there? " << omp_get_num_threads() << std::endl << std::flush;
#pragma omp parallel num_threads(3)
#pragma omp master
{
std::cout << "What is my thread id? " << omp_get_thread_num() << std::endl << std::flush;
}
std::cout << "How many threads are there? " << omp_get_num_threads() << std::endl << std::flush;

return 0;
}
