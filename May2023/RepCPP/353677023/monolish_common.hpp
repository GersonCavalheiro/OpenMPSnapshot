#pragma once
#if USE_SXAT
#undef _HAS_CPP17
#endif
#include "monolish_dense.hpp"
#include "monolish_logger.hpp"
#include "monolish_matrix.hpp"
#include "monolish_tensor.hpp"
#include "monolish_vector.hpp"
#include "monolish_view1D.hpp"
#include <initializer_list>

#define MONOLISH_SOLVER_SUCCESS 0
#define MONOLISH_SOLVER_SIZE_ERROR -1
#define MONOLISH_SOLVER_MAXITER -2
#define MONOLISH_SOLVER_BREAKDOWN -3
#define MONOLISH_SOLVER_RESIDUAL_NAN -4
#define MONOLISH_SOLVER_NOT_IMPL -10


namespace monolish {

namespace util {



int get_num_devices();


bool set_default_device(size_t device_num);


int get_default_device();




double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);

float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);


double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);

float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
float get_residual_l2(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);


double get_residual_l2(const matrix::LinearOperator<double> &A,
const vector<double> &x, const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const vector<double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
double get_residual_l2(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);




[[nodiscard]] bool solver_check(const int err);



void set_log_level(const size_t Level);


void set_log_filename(const std::string filename);



template <typename T>
void random_vector(vector<T> &vec, const T min, const T max);


template <typename T>
void random_vector(vector<T> &vec, const T min, const T max,
const std::uint32_t seed);



template <typename T, typename U>
[[nodiscard]] bool is_same_structure(const T A, const U B) {
return false;
}


template <typename T>
[[nodiscard]] bool is_same_structure(const vector<T> &x, const vector<T> &y) {
return x.size() == y.size();
}


template <typename T>
[[nodiscard]] bool is_same_structure(const matrix::Dense<T> &A,
const matrix::Dense<T> &B);


template <typename T>
[[nodiscard]] bool is_same_structure(const matrix::COO<T> &A,
const matrix::COO<T> &B);


template <typename T>
[[nodiscard]] bool is_same_structure(const matrix::CRS<T> &A,
const matrix::CRS<T> &B);


template <typename T>
bool is_same_structure(const matrix::LinearOperator<T> &A,
const matrix::LinearOperator<T> &B);


template <typename T>
[[nodiscard]] bool is_same_structure(const tensor::tensor_Dense<T> &A,
const tensor::tensor_Dense<T> &B);


template <typename T>
[[nodiscard]] bool is_same_structure(const tensor::tensor_COO<T> &A,
const tensor::tensor_COO<T> &B);


template <typename T, typename... types>
[[nodiscard]] bool is_same_structure(const T &A, const T &B,
const types &...args) {
return is_same_structure(A, B) && is_same_structure(A, args...);
}


template <typename T, typename U>
[[nodiscard]] bool is_same_size(const T &x, const U &y) {
return x.size() == y.size();
}


template <typename T>
[[nodiscard]] bool is_same_size(const matrix::Dense<T> &A,
const matrix::Dense<T> &B);


template <typename T>
[[nodiscard]] bool is_same_size(const matrix::COO<T> &A,
const matrix::COO<T> &B);


template <typename T>
[[nodiscard]] bool is_same_size(const matrix::CRS<T> &A,
const matrix::CRS<T> &B);


template <typename T>
[[nodiscard]] bool is_same_size(const matrix::LinearOperator<T> &A,
const matrix::LinearOperator<T> &B);


template <typename T>
[[nodiscard]] bool is_same_size(const tensor::tensor_Dense<T> &A,
const tensor::tensor_Dense<T> &B);


template <typename T>
[[nodiscard]] bool is_same_size(const tensor::tensor_COO<T> &A,
const tensor::tensor_COO<T> &B);


template <typename T, typename U, typename... types>
[[nodiscard]] bool is_same_size(const T &arg1, const U &arg2,
const types &...args) {
return is_same_size(arg1, arg2) && is_same_size(arg1, args...);
}


template <typename T, typename U>
[[nodiscard]] bool is_same_device_mem_stat(const T &arg1, const U &arg2) {
return arg1.get_device_mem_stat() == arg2.get_device_mem_stat();
}


template <typename T, typename U, typename... types>
[[nodiscard]] bool is_same_device_mem_stat(const T &arg1, const U &arg2,
const types &...args) {
return is_same_device_mem_stat(arg1, arg2) &&
is_same_device_mem_stat(arg1, args...);
}



template <typename T>
[[nodiscard]] matrix::COO<T> band_matrix(const int M, const int N, const int W,
const T diag_val, const T val);


template <typename T>
[[nodiscard]] matrix::COO<T> asym_band_matrix(const int M, const int N,
const int W, const T diag_val,
const T Uval, const T Lval);


template <typename T>
[[nodiscard]] matrix::COO<T> random_structure_matrix(const int M, const int N,
const int nnzrow,
const T val);


template <typename T>
[[nodiscard]] tensor::tensor_COO<T>
random_structure_tensor(const size_t M, const size_t N, const size_t nnzrow,
const T val);


template <typename T>
[[nodiscard]] tensor::tensor_COO<T>
random_structure_tensor(const size_t M, const size_t N, const size_t L,
const size_t nnzrow, const T val);


template <typename T>
[[nodiscard]] tensor::tensor_COO<T>
random_structure_tensor(const size_t K, const size_t M, const size_t N,
const size_t L, const size_t nnzrow, const T val);


template <typename T> [[nodiscard]] matrix::COO<T> eye(const int M);


template <typename T> [[nodiscard]] matrix::COO<T> frank_matrix(const int &M);


template <typename T>
[[nodiscard]] T frank_matrix_eigenvalue(const int &M, const int &N);


template <typename T>
[[nodiscard]] matrix::COO<T> tridiagonal_toeplitz_matrix(const int &M, T a,
T b);


template <typename T>
[[nodiscard]] T tridiagonal_toeplitz_matrix_eigenvalue(const int &M, int N, T a,
T b);


template <typename T>
[[nodiscard]] matrix::COO<T> laplacian_matrix_1D(const int &M);


template <typename T>
[[nodiscard]] T laplacian_matrix_1D_eigenvalue(const int &M, int N);


template <typename T>
[[nodiscard]] matrix::COO<T> laplacian_matrix_2D_5p(const int M, const int N);


template <typename T>
[[nodiscard]] matrix::COO<T> toeplitz_plus_hankel_matrix(const int &M, T a0,
T a1, T a2);


template <typename T>
[[nodiscard]] T toeplitz_plus_hankel_matrix_eigenvalue(const int &M, int N,
T a0, T a1, T a2, T b0,
T b1, T b2);



template <typename T> void send(T &x) { x.send(); }


template <typename T, typename... Types> void send(T &x, Types &...args) {
x.send();
send(args...);
}


template <typename T> void recv(T &x) { x.recv(); }


template <typename T, typename... Types> void recv(T &x, Types &...args) {
x.recv();
recv(args...);
}



template <typename T> void device_free(T &x) { x.device_free(); }


template <typename T, typename... Types>
void device_free(T &x, Types &...args) {
x.device_free();
device_free(args...);
}


[[nodiscard]] bool build_with_avx();


[[nodiscard]] bool build_with_avx2();


[[nodiscard]] bool build_with_avx512();


[[nodiscard]] bool build_with_mpi();


[[nodiscard]] bool build_with_gpu();


[[nodiscard]] bool build_with_mkl();


[[nodiscard]] bool build_with_lapack();


[[nodiscard]] bool build_with_cblas();


} 
} 
