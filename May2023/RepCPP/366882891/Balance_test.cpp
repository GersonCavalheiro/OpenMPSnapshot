#include <iostream>
#include <sstream>
#include <omp.h>
#include <vector>
#include <cmath>


int func(int x) {
return pow(x, 3);
}


int main()
{
int THREADS = omp_get_num_procs();

int N = 65;


#pragma omp parallel num_threads(THREADS)
{
int thread_num = omp_get_thread_num();

#pragma omp for schedule(dynamic, 4)
for (int i = 0; i < N; ++i) {
std::stringstream buf;
buf << "Thread " << thread_num << ", Iteration: " << i << std::endl;
std::cout << buf.str();

func(i);
}               
}

std::cout << "Main thread completed task" << std::endl;
}

