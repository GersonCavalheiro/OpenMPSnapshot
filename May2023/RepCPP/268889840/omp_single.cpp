#include <iostream>
#include <omp.h>
#include <unistd.h> 
#include <vector>

int main()
{
int n_threads, thread_id;

double value = 5.0;
int size = 10;
std::vector<double> a(size), b(size), c(size);
std::fill(a.begin(), a.end(), 0.6*value);
std::fill(b.begin(), b.end(), 0.4*value);
std::fill(c.begin(), c.end(), 0.0);

#pragma omp parallel private(n_threads, thread_id)
{
#pragma omp single
{
n_threads = omp_get_num_threads();
std::cout << "Parallel Region Start ..." << std::endl;
std::cout << "Number of threads = " << n_threads << std::endl;
}
thread_id = omp_get_thread_num();
usleep(5000 * thread_id);
#pragma omp for
for (int i = 0; i < size; i++)
{
c[i] = a[i]+b[i];
std::cout << "Thread " << thread_id << " executes iteration " << i << "."<< std::endl;
}
}
}