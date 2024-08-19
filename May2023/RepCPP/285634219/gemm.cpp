



#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <vector>
#include <omp.h>
#include "mkl.h"
#include "mkl_omp_offload.h"

template <typename T>
void print_2x2_matrix_values(T M, int ldM, std::string M_name) 
{
std::cout << std::endl;
std::cout << "\t\t\t" << M_name << " = [ " << M[0*ldM + 0] << ", " << M[1*ldM + 0]         << ", ...\n";
std::cout << "\t\t\t    [ "                << M[0*ldM + 1] << ", " << M[1*ldM + 1] << ", ...\n";
std::cout << "\t\t\t    [ "                << "...\n";
std::cout << std::endl;
}

template <typename fp> void rand_matrix(fp *M, int n_row, int n_col)
{
for (int i = 0; i < n_row; i++)
for (int j = 0; j < n_col; j++)
M[i * n_col + j] = rand() % 5;
}

template <typename fp>
void run_gemm_example(int repeat) {

MKL_INT m = 79;
MKL_INT n = 83; 
MKL_INT k = 91;

fp alpha = fp(2.0); 
fp beta  = fp(0.5);

fp* a = (float *)mkl_malloc((m * k) * sizeof(float), 64);
fp* b = (float *)mkl_malloc((k * n) * sizeof(float), 64);
fp* c = (float *)mkl_malloc((m * n) * sizeof(float), 64);

srand(2);
rand_matrix(a, m, k);
rand_matrix(b, k, n);
rand_matrix(c, m, n);


#pragma omp target data map(to:a[0:m*k], b[0:k*n]) map(tofrom:c[0:m*n]) device(0)
{
auto start = std::chrono::steady_clock::now();

for (int i = 0; i < repeat; i++) 
{
#pragma omp target variant dispatch device(0) use_device_ptr(a, b, c)
sgemm("N", "N", &n, &m, &k, &alpha, b, &n, a, &k, &beta, c, &n);
}

auto end = std::chrono::steady_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
printf("Average sgemm execution time: %f (s)\n", (time * 1e-9f) / repeat);
}

std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

print_2x2_matrix_values(a, k, "A");

print_2x2_matrix_values(b, n, "B");

print_2x2_matrix_values(c, n, "C");

mkl_free(a);
mkl_free(b);
mkl_free(c);
}

int main (int argc, char ** argv) {
if (argc != 2) {
printf("Usage: %s <repeat>\n", argv[0]);
return 1;
}
const int repeat = atoi(argv[1]);

std::cout << "\tRunning with single precision real data type:" << std::endl;
run_gemm_example<float>(repeat);
return 0;
}
