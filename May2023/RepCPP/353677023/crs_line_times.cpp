#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename T, typename VEC>
void times_row_core(const matrix::CRS<T> &A, const size_t num, const VEC &x,
matrix::CRS<T> &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_structure(A, C));
assert(util::is_same_device_mem_stat(A, x, C));
assert(A.get_col() == x.size());
assert(A.get_row() >= num);

const auto *Ad = A.data();
auto *Cd = C.data();
const auto *rowd = A.row_ptr.data();
const auto *cold = A.col_ind.data();

const auto *xd = x.begin();

if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto j = rowd[num]; j < rowd[num + 1]; j++) {
Cd[j] = Ad[j] * xd[cold[j]];
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto j = rowd[num]; j < rowd[num + 1]; j++) {
Cd[j] = Ad[j] * xd[cold[j]];
}
}

logger.func_out();
}

template <typename T, typename VEC>
void times_col_core(const matrix::CRS<T> &A, const size_t num, const VEC &x,
matrix::CRS<T> &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_structure(A, C));
assert(util::is_same_device_mem_stat(A, x, C));
assert(A.get_row() == x.size());
assert(A.get_col() >= num);

const auto *Ad = A.data();
auto *Cd = C.data();
const auto *rowd = A.row_ptr.data();
const auto *cold = A.col_ind.data();
const auto n = A.get_row();

const auto *xd = x.begin();

if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
for (auto i = decltype(n){0}; i < n; i++) {
for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
if (cold[j] == num) {
Cd[j] = Ad[j] * xd[i];
}
}
}
#else
throw std::runtime_error(
"error USE_GPU is false, but get_device_mem_stat() == true");
#endif
} else {
#pragma omp parallel for
for (auto i = decltype(n){0}; i < n; i++) {
for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
if (cold[j] == num) {
Cd[j] = Ad[j] * xd[i];
}
}
}
}

logger.func_out();
}
} 

namespace blas {
void times_row(const matrix::CRS<double> &A, const size_t num,
const vector<double> &x, matrix::CRS<double> &C) {
times_row_core(A, num, x, C);
}
void times_row(const matrix::CRS<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
matrix::CRS<double> &C) {
times_row_core(A, num, x, C);
}
void times_row(const matrix::CRS<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
matrix::CRS<double> &C) {
times_row_core(A, num, x, C);
}
void times_row(const matrix::CRS<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::CRS<double> &C) {
times_row_core(A, num, x, C);
}
void times_row(const matrix::CRS<float> &A, const size_t num,
const vector<float> &x, matrix::CRS<float> &C) {
times_row_core(A, num, x, C);
}
void times_row(const matrix::CRS<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::CRS<float> &C) {
times_row_core(A, num, x, C);
}
void times_row(const matrix::CRS<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
matrix::CRS<float> &C) {
times_row_core(A, num, x, C);
}
void times_row(const matrix::CRS<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::CRS<float> &C) {
times_row_core(A, num, x, C);
}

void times_col(const matrix::CRS<double> &A, const size_t num,
const vector<double> &x, matrix::CRS<double> &C) {
times_col_core(A, num, x, C);
}
void times_col(const matrix::CRS<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
matrix::CRS<double> &C) {
times_col_core(A, num, x, C);
}
void times_col(const matrix::CRS<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
matrix::CRS<double> &C) {
times_col_core(A, num, x, C);
}
void times_col(const matrix::CRS<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::CRS<double> &C) {
times_col_core(A, num, x, C);
}
void times_col(const matrix::CRS<float> &A, const size_t num,
const vector<float> &x, matrix::CRS<float> &C) {
times_col_core(A, num, x, C);
}
void times_col(const matrix::CRS<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::CRS<float> &C) {
times_col_core(A, num, x, C);
}
void times_col(const matrix::CRS<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
matrix::CRS<float> &C) {
times_col_core(A, num, x, C);
}
void times_col(const matrix::CRS<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::CRS<float> &C) {
times_col_core(A, num, x, C);
}

} 
} 
