#include <iostream>
#include <omp.h>
#include <unistd.h> 
#include <vector>

int main()
{
int n_threads, thread_id;

int size = omp_get_max_threads();
std::vector<double> v(size), w(size), k(size);
std::fill(v.begin(), v.end(), 1.0);
std::fill(w.begin(), w.end(), 1.0);
std::fill(k.begin(), k.end(), 0.0);

double value{10.0};

#pragma omp parallel private(n_threads, thread_id) shared(v, w, k, value)
{
thread_id = omp_get_thread_num();
v[thread_id] += value;
w[thread_id] += value;

#pragma omp flush

#pragma omp for
for (int i = 0; i < size; i++)
{
k[i] = v[i] + w[i];
}
}
for  (int i = 0; i < v.size(); i++)
std::cout << "k[" << i << "] = " << k[i] << std::endl;
}