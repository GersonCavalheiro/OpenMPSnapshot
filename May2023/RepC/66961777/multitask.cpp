#include <iostream>
#include <omp.h>
#include <math.h>
const double PI = 3.141592653589793238463;
using namespace std;
int num_threads, upper_bound;
bool isPrime(int num) {
for (int i=2; i<num; i++) {
if (num % i == 0) {
return false;
}
}
return true;
}
int* prime_table(int num) {
int* primes = new int[num]; 
int prime_number = 2, i = 0;
while (i < num) {
if (isPrime(prime_number)) {
primes[i] = prime_number;
i++;
}
prime_number++;
}
return primes;
}
double* sine_table(int num) {
double* sines = new double[num];
double a;
#pragma omp parallel for private(a) num_threads(num_threads)
for (int i=0; i<num; i++) {
sines[i] = 0.0;
for (int j=0; j <= i; j++) {
a = j * PI / (num - 1);
sines[i] += sin(a);
}
}
return sines;
}
int main(int argc, char* argv[]) {
if (argc < 3) {
cout << "Usage: ./multitask.o upper_bound num_threads" << endl;
return 1;
}
upper_bound = atoi(argv[1]);
num_threads = atoi(argv[2]);
int* primes;
double* sines;
double start_time = omp_get_wtime();
double task1_start, task2_start, task1_end, task2_end;
#pragma omp parallel sections num_threads(num_threads)
{
#pragma omp section
{
task1_start = omp_get_wtime();
primes = prime_table(upper_bound);
task1_end = omp_get_wtime();
}
#pragma omp section
{
task2_start = omp_get_wtime();
sines = sine_table(upper_bound);
task2_end = omp_get_wtime();
}
}
double end_time = omp_get_wtime();
cout << "Largest Prime: " << primes[upper_bound - 1] << endl;
cout << "Largest Sine: " << sines[upper_bound - 1] << endl;
cout << "Total Time: " << end_time - start_time << "s" << endl;
cout << "Task 1 Time: " << task1_end - task1_start << "s" << endl;
cout << "Task 2 Time: " << task2_end - task2_start << "s" << endl;
return 0;
}
