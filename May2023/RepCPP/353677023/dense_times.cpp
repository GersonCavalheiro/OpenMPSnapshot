#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename T>
void times_core(const T alpha, const matrix::Dense<T> &A, matrix::Dense<T> &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_structure(A, C));
assert(util::is_same_device_mem_stat(A, C));

internal::vmul(A.get_nnz(), A.data(), alpha, C.data(),
A.get_device_mem_stat());

logger.func_out();
}

template <typename T, typename VEC>
void times_row_core(const matrix::Dense<T> &A, const VEC &x,
matrix::Dense<T> &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_structure(A, C));
assert(util::is_same_device_mem_stat(A, x, C));
assert(A.get_col() == x.size());

const auto *Ad = A.data();
const auto m = A.get_row();
const auto n = A.get_col();
auto *Cd = C.data();

const auto *xd = x.begin();

if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(m){0}; i < m; i++) {
for (auto j = decltype(n){0}; j < n; j++) {
Cd[i * n + j] = Ad[i * n + j] * xd[j];
}
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(m){0}; i < m; i++) {
for (auto j = decltype(n){0}; j < n; j++) {
Cd[i * n + j] = Ad[i * n + j] * xd[j];
}
}
}

logger.func_out();
}

template <typename T, typename VEC>
void times_col_core(const matrix::Dense<T> &A, const VEC &x,
matrix::Dense<T> &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_structure(A, C));
assert(util::is_same_device_mem_stat(A, x, C));
assert(A.get_row() == x.size());

const auto *Ad = A.data();
const auto m = A.get_row();
const auto n = A.get_col();
auto *Cd = C.data();

const auto *xd = x.begin();

if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(m){0}; i < m; i++) {
for (auto j = decltype(n){0}; j < n; j++) {
Cd[i * n + j] = Ad[i * n + j] * xd[i];
}
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(m){0}; i < m; i++) {
for (auto j = decltype(n){0}; j < n; j++) {
Cd[i * n + j] = Ad[i * n + j] * xd[i];
}
}
}

logger.func_out();
}
} 

namespace blas {

void times(const double alpha, const matrix::Dense<double> &A,
matrix::Dense<double> &C) {
times_core(alpha, A, C);
}

void times(const float alpha, const matrix::Dense<float> &A,
matrix::Dense<float> &C) {
times_core(alpha, A, C);
}

void times_row(const matrix::Dense<double> &A, const vector<double> &x,
matrix::Dense<double> &C) {
times_row_core(A, x, C);
}
void times_row(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C) {
times_row_core(A, x, C);
}
void times_row(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C) {
times_row_core(A, x, C);
}
void times_row(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C) {
times_row_core(A, x, C);
}
void times_row(const matrix::Dense<float> &A, const vector<float> &x,
matrix::Dense<float> &C) {
times_row_core(A, x, C);
}
void times_row(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C) {
times_row_core(A, x, C);
}
void times_row(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C) {
times_row_core(A, x, C);
}
void times_row(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C) {
times_row_core(A, x, C);
}

void times_col(const matrix::Dense<double> &A, const vector<double> &x,
matrix::Dense<double> &C) {
times_col_core(A, x, C);
}
void times_col(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C) {
times_col_core(A, x, C);
}
void times_col(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C) {
times_col_core(A, x, C);
}
void times_col(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C) {
times_col_core(A, x, C);
}
void times_col(const matrix::Dense<float> &A, const vector<float> &x,
matrix::Dense<float> &C) {
times_col_core(A, x, C);
}
void times_col(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C) {
times_col_core(A, x, C);
}
void times_col(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C) {
times_col_core(A, x, C);
}
void times_col(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C) {
times_col_core(A, x, C);
}

} 
} 
