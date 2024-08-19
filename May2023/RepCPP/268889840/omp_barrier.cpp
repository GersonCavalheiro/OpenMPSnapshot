#include <iostream>
#include <omp.h>
#include <unistd.h> 
#include <stdlib.h>

int main()
{
int num_threads, thread_id;

#pragma omp parallel private(thread_id)
{
num_threads = omp_get_num_threads();
thread_id = omp_get_thread_num();

if( thread_id < num_threads/2)
system("sleep 3");
std::cout << "Before Barrier: thread #" << thread_id << std::endl;
#pragma omp barrier
std::cout << "After Barrier: thread #" << thread_id << std::endl;
}
}