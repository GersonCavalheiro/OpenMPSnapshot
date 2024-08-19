#include <iostream>
#include <chrono>
#include <intrin.h>
#include <cmath>
#include <omp.h>

int main() {
const float Q = 9;
int K;
int N;

std::cout << "Enter matrix dimensions:\n";
std::cout << "K = ";
std::cin >> K;
std::cout << "N = ";
std::cin >> N;

long double result = 1.0;
double** arr = new double* [K];

auto start_t = std::chrono::high_resolution_clock::now();
auto start_c = __rdtsc();
#pragma omp parallel 
{
double* sum = new double[K];
long double prod = 1.0;
#pragma omp for
for (int i = 1; i < K; i++) {
sum[i] = 0.0;
arr[i] = new double[N];
for (int j = 1; j < N; j++) {
arr[i][j] = sqrt(Q * double(j) / 100.0);
sum[i] += arr[i][j];
}
prod *= sum[i];
delete[] arr[i];
}
#pragma omp atomic
result *= prod;
delete[] sum;
}
auto end_c = __rdtsc();
auto end_t = std::chrono::high_resolution_clock::now();

std::chrono::duration<float> time = end_t - start_t;

std::cout << "\nExecution result: " << result << '\n';
std::cout << "Execution clocks: " << end_c - start_c << '\n';
std::cout << "Execution time: " << time.count() << " sec\n";

delete[] arr;

system("pause");

return 0;
}