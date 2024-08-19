#include <iostream>
#include <sstream>
#include <omp.h>
#include <vector>

int main()
{
int THREADS = omp_get_num_procs();

int N = 2000;
double sum = 0;

#pragma omp parallel num_threads(THREADS) reduction(+ : sum)
{
int thread_num = omp_get_thread_num();
sum = 0;

#pragma omp for
for (int i = 1; i <= N; ++i) {
sum += 1 / double(i);
}


std::stringstream buf;
buf << "Thread " << thread_num << " completed task. Result: " << sum << std::endl;
std::cout << buf.str();
}

std::cout << "Main thread completed task. Result: " << sum << std::endl;
}

