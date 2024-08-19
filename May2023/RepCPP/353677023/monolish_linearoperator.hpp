#pragma once
#include <exception>
#include <functional>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#if USE_SXAT
#undef _HAS_CPP17
#endif
#include <random>
#if USE_SXAT
#define _HAS_CPP17 1
#endif

namespace monolish {
template <typename Float> class vector;
namespace matrix {
template <typename Float> class Dense;
template <typename Float> class COO;
template <typename Float> class CRS;




template <typename Float> class LinearOperator {
private:

size_t rowN;


size_t colN;


mutable bool gpu_status = false;


std::function<vector<Float>(const vector<Float> &)> matvec = nullptr;


std::function<vector<Float>(const vector<Float> &)> rmatvec = nullptr;


std::function<Dense<Float>(const Dense<Float> &)> matmul_dense = nullptr;


std::function<Dense<Float>(const Dense<Float> &)> rmatmul_dense = nullptr;

public:
LinearOperator() {}


LinearOperator(const size_t M, const size_t N);


LinearOperator(
const size_t M, const size_t N,
const std::function<vector<Float>(const vector<Float> &)> &MATVEC);


LinearOperator(
const size_t M, const size_t N,
const std::function<vector<Float>(const vector<Float> &)> &MATVEC,
const std::function<vector<Float>(const vector<Float> &)> &RMATVEC);


LinearOperator(
const size_t M, const size_t N,
const std::function<Dense<Float>(const Dense<Float> &)> &MATMUL);


LinearOperator(
const size_t M, const size_t N,
const std::function<Dense<Float>(const Dense<Float> &)> &MATMUL,
const std::function<Dense<Float>(const Dense<Float> &)> &RMATMUL);


void convert(COO<Float> &coo);


LinearOperator(COO<Float> &coo) { convert(coo); }

void convert(CRS<Float> &crs);

LinearOperator(CRS<Float> &crs) { convert(crs); }

void convert(Dense<Float> &dense);

LinearOperator(Dense<Float> &dense) { convert(dense); }

void convert_to_Dense(Dense<Float> &dense) const;


LinearOperator(const LinearOperator<Float> &linearoperator);


[[nodiscard]] size_t get_row() const { return rowN; }


[[nodiscard]] size_t get_col() const { return colN; }


[[nodiscard]] std::function<vector<Float>(const vector<Float> &)>
get_matvec() const {
return matvec;
}


[[nodiscard]] std::function<vector<Float>(const vector<Float> &)>
get_rmatvec() const {
return rmatvec;
}


[[nodiscard]] std::function<
matrix::Dense<Float>(const matrix::Dense<Float> &)>
get_matmul_dense() const {
return matmul_dense;
}


[[nodiscard]] std::function<
matrix::Dense<Float>(const matrix::Dense<Float> &)>
get_rmatmul_dense() const {
return rmatmul_dense;
}


[[nodiscard]] bool get_matvec_init_flag() const {
return !(matvec == nullptr);
}


[[nodiscard]] bool get_rmatvec_init_flag() const {
return !(rmatvec == nullptr);
}


[[nodiscard]] bool get_matmul_dense_init_flag() const {
return !(matmul_dense == nullptr);
}


[[nodiscard]] bool get_rmatmul_dense_init_flag() const {
return !(rmatmul_dense == nullptr);
}


void
set_matvec(const std::function<vector<Float>(const vector<Float> &)> &MATVEC);


void set_rmatvec(
const std::function<vector<Float>(const vector<Float> &)> &RMATVEC);


void set_matmul_dense(
const std::function<matrix::Dense<Float>(const matrix::Dense<Float> &)>
&MATMUL);


void set_rmatmul_dense(
const std::function<matrix::Dense<Float>(const matrix::Dense<Float> &)>
&RMATMUL);


[[nodiscard]] std::string type() const { return "LinearOperator"; }


void send() const {};


void recv() const {};


void nonfree_recv() const {};


void device_free() const {};


[[nodiscard]] bool get_device_mem_stat() const { return gpu_status; };

void set_device_mem_stat(bool status) {
gpu_status = status;
return;
};


~LinearOperator() {}


void diag(vector<Float> &vec) const;
void diag(view1D<vector<Float>, Float> &vec) const;
void diag(view1D<matrix::Dense<Float>, Float> &vec) const;
void diag(view1D<tensor::tensor_Dense<Float>, Float> &vec) const;


void operator=(const LinearOperator<Float> &mat);
};


} 
} 
