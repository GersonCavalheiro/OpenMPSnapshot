#include <iostream>
#include <omp.h>
#include <unistd.h> 
#include <vector>

void sum(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c){
for (int i = 0; i < a.size(); i++) {
c[i] = a[i]+b[i];
}
}

void sub(std::vector<double> &a, std::vector<double> &b, std::vector<double> &d){
for (int i = 0; i < a.size(); i++) {
d[i] = a[i]-b[i];
}
}

int main()
{
int n_threads, thread_id;

double value = 5.0;
int size = 40;
std::vector<double> a(size), b(size), c(size), d(size);
std::fill(a.begin(), a.end(), 0.6*value);
std::fill(b.begin(), b.end(), 0.4*value);
std::fill(c.begin(), c.end(), 0.0);
std::fill(d.begin(), d.end(), 0.0);

n_threads = omp_get_max_threads();
std::cout << "Max number of threads: " << n_threads << std::endl;

#pragma omp parallel private(thread_id)
{
thread_id = omp_get_thread_num();
#pragma omp sections
{
#pragma omp section
{
usleep(5000 * thread_id);
std::cout << "Thread " << thread_id << " executes SUM." << std::endl;
sum(a,b,c);
}
#pragma omp section
{
usleep(5000 * thread_id);
std::cout << "Thread " << thread_id << " executes SUBTRACTION." << std::endl;
sub(a,b,d);
}
}
}
}