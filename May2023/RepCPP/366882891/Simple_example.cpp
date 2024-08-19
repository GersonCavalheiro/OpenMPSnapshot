#include <iostream>
#include <sstream>
#include <omp.h>


int x = 0;
int y = 0;

#pragma omp threadprivate(x, y)

int main()
{
int THREADS = omp_get_num_procs();

x = 10;
y = 10;

std::cout << "Main thread: x = " << x << ", y = " << y << std::endl;


#pragma omp parallel num_threads(THREADS) copyin(y)
{

int thread_num = omp_get_thread_num();

x += thread_num + 1;
y += thread_num + 1;

std::stringstream buf;
buf << "Thread " << thread_num << ": x = " << x << ", y = " << y << std::endl;
std::cout << buf.str();
}

std::cout << "Main thread: x = " << x << ", y = " << y << std::endl;
}

