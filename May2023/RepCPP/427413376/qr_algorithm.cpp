#include <chrono>
#include <numeric>
#include <vector>
#include <cassert>
#include <cmath>
#include <omp.h>
#include <string>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>

template<typename T = double>
struct Matrix {
explicit Matrix(int n, T init = 0) : n(n), v(n * n, init) {}

int n;
std::vector<T> v;
};

template<typename T>
Matrix<T> load_matrix(const std::string &path) {
std::ifstream file(path);
if (!file.is_open())
throw std::exception("Failed to open file");

int m, n, n_values;
file >> m >> n >> n_values;
assert(m == n);

Matrix<T> M(n);

for (int k = 0; k < n_values; k++) {
int i, j;
T v;
file >> i >> j >> v;
i -= 1;
j -= 1;
assert(0 <= i && i < n);
assert(0 <= j && j < n);
M.v[i * n + j] = v;
}

return std::move(M);
}

template<typename T>
void print_matrix(const std::string &name, const Matrix<T> &M, int width = 8, int precision = 5) {
std::cout << std::setprecision(precision) << "Matrix " << name << std::endl;
for (int i = 0; i < M.n; i++) {
for (int j = 0; j < M.n; j++) {
std::cout << std::setw(width) << M.v[i * M.n + j] << " ";
}
std::cout << std::endl;
}
}

template<typename T>
void qr_decomposition(const Matrix<T> &A ,
Matrix<T> &Q ,
Matrix<T> &R ) {
assert(A.n == Q.n);
auto n = A.n;

Q = A;

#pragma omp parallel for
for (int i = 0; i < n; i++)
for (int k = 0; k < n; k++)
R.v[i * n + k] = 0;

for (int i = 0; i < n; i++)
R.v[i * n + i] = 1;

for (int i = 0; i < n; i++) {
T norm = 0;
for (int k = 0; k < n; k++)
norm += Q.v[i * n + k] * Q.v[i * n + k];
norm = std::sqrt(norm);
for (int k = 0; k < n; k++)
Q.v[i * n + k] /= norm;
R.v[i * n + i] *= norm;


#pragma omp parallel for
for (int j = i + 1; j < n; j++) {
T q_i_dot_q_j = 0;
for (int k = 0; k < n; k++)
q_i_dot_q_j += Q.v[i * n + k] * Q.v[j * n + k];
for (int k = 0; k < n; k++)
Q.v[j * n + k] -= q_i_dot_q_j * Q.v[i * n + k];
R.v[i * n + j] += q_i_dot_q_j;
}
}
}

template<typename T>
void multiply(const Matrix<T> &A ,
const Matrix<T> &B ,
Matrix<T> &C ) {
assert(A.n == B.n);
assert(A.n == C.n);
auto n = A.n;

#pragma omp parallel for
for (int j = 0; j < n; j++) {
for (int i = 0; i < n; i++) {
T sum = 0;
for (int k = 0; k < n; k++)
sum += A.v[i * n + k] * B.v[j * n + k];
C.v[j * n + i] = sum;
}
}
}

template<typename T>
std::vector<T> qr_algorithm(Matrix<T> &A, int iterations = 10) {
Matrix<T> X = A;  
Matrix<T> Q(A.n); 
Matrix<T> R(A.n); 

for (int k = 0; k < iterations; k++) {
qr_decomposition(X, Q, R);
multiply(R, Q, X);
}

std::vector<T> eigenvalues;
eigenvalues.reserve(X.n);

for (int i = 0; i < X.n; i++)
eigenvalues.push_back(X.v[i * X.n + i]);

return eigenvalues;
}

int main(int argc, const char *const *argv) {
if (argc != 5) {
std::cerr << "Arguments count mismatched";
return -1;
}

using T = double;
using clock = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

int samples = std::atoi(argv[1]);
int threads_count = std::atoi(argv[2]);
int iterations = std::atoi(argv[3]);
std::string matrix_path = argv[4];
std::vector<double> times;

omp_set_num_threads(threads_count);

Matrix<T> A = load_matrix<T>(matrix_path); 

for (int i = 0; i < samples; i++) {
auto time_point = clock::now();
qr_algorithm(A, iterations);
times.push_back(static_cast<double >(std::chrono::duration_cast<ns>(clock::now() - time_point).count()) / 1e9f);
std::cout << i << " ";
}

double average = std::reduce(times.begin(), times.end(), 0.0) / static_cast<double>(samples);
double min_time = *std::min_element(times.begin(), times.end());
double max_time = *std::max_element(times.begin(), times.end());
double sd = std::transform_reduce(times.begin(), times.end(), 0.0, std::plus<>(),
[=](auto x) { return (x - average) * (x - average); })
/ static_cast<double>(samples - 1);
std::cout << "res: "
<< average << ", "
<< min_time << ", "
<< max_time << ", "
<< sd << " (all in sec)" << std::endl;

return 0;
}