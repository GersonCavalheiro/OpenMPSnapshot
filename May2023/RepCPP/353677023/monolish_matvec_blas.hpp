#pragma once
#include "../common/monolish_common.hpp"

namespace monolish {

namespace blas {





void times_row(const matrix::Dense<double> &A, const vector<double> &x,
matrix::Dense<double> &C);
void times_row(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void times_row(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_row(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_row(const matrix::Dense<float> &A, const vector<float> &x,
matrix::Dense<float> &C);
void times_row(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void times_row(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void times_row(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void times_row(const matrix::Dense<double> &A, const size_t num,
const vector<double> &x, matrix::Dense<double> &C);
void times_row(const matrix::Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void times_row(const matrix::Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_row(const matrix::Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_row(const matrix::Dense<float> &A, const size_t num,
const vector<float> &x, matrix::Dense<float> &C);
void times_row(const matrix::Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void times_row(const matrix::Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void times_row(const matrix::Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void times_row(const matrix::CRS<double> &A, const vector<double> &x,
matrix::CRS<double> &C);
void times_row(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x, matrix::CRS<double> &C);
void times_row(const matrix::CRS<double> &A,
const view1D<matrix::CRS<double>, double> &x,
matrix::CRS<double> &C);
void times_row(const matrix::CRS<float> &A, const vector<float> &x,
matrix::CRS<float> &C);
void times_row(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x, matrix::CRS<float> &C);
void times_row(const matrix::CRS<float> &A,
const view1D<matrix::CRS<float>, float> &x,
matrix::CRS<float> &C);




void times_row(const matrix::CRS<double> &A, const size_t num,
const vector<double> &x, matrix::CRS<double> &C);
void times_row(const matrix::CRS<double> &A, const size_t num,
const view1D<vector<double>, double> &x, matrix::CRS<double> &C);
void times_row(const matrix::CRS<double> &A, const size_t num,
const view1D<matrix::CRS<double>, double> &x,
matrix::CRS<double> &C);
void times_row(const matrix::CRS<float> &A, const size_t num,
const vector<float> &x, matrix::CRS<float> &C);
void times_row(const matrix::CRS<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::CRS<float> &C);
void times_row(const matrix::CRS<float> &A, const size_t num,
const view1D<matrix::CRS<float>, float> &x,
matrix::CRS<float> &C);




void times_col(const matrix::Dense<double> &A, const vector<double> &x,
matrix::Dense<double> &C);
void times_col(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void times_col(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_col(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_col(const matrix::Dense<float> &A, const vector<float> &x,
matrix::Dense<float> &C);
void times_col(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void times_col(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void times_col(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void times_col(const matrix::Dense<double> &A, const size_t num,
const vector<double> &x, matrix::Dense<double> &C);
void times_col(const matrix::Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void times_col(const matrix::Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_col(const matrix::Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void times_col(const matrix::Dense<float> &A, const size_t num,
const vector<float> &x, matrix::Dense<float> &C);
void times_col(const matrix::Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void times_col(const matrix::Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void times_col(const matrix::Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void times_col(const matrix::CRS<double> &A, const vector<double> &x,
matrix::CRS<double> &C);
void times_col(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x, matrix::CRS<double> &C);
void times_col(const matrix::CRS<double> &A,
const view1D<matrix::CRS<double>, double> &x,
matrix::CRS<double> &C);
void times_col(const matrix::CRS<float> &A, const vector<float> &x,
matrix::CRS<float> &C);
void times_col(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x, matrix::CRS<float> &C);
void times_col(const matrix::CRS<float> &A,
const view1D<matrix::CRS<float>, float> &x,
matrix::CRS<float> &C);




void times_col(const matrix::CRS<double> &A, const size_t num,
const vector<double> &x, matrix::CRS<double> &C);
void times_col(const matrix::CRS<double> &A, const size_t num,
const view1D<vector<double>, double> &x, matrix::CRS<double> &C);
void times_col(const matrix::CRS<double> &A, const size_t num,
const view1D<matrix::CRS<double>, double> &x,
matrix::CRS<double> &C);
void times_col(const matrix::CRS<float> &A, const size_t num,
const vector<float> &x, matrix::CRS<float> &C);
void times_col(const matrix::CRS<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::CRS<float> &C);
void times_col(const matrix::CRS<float> &A, const size_t num,
const view1D<matrix::CRS<float>, float> &x,
matrix::CRS<float> &C);




void adds_row(const matrix::Dense<double> &A, const vector<double> &x,
matrix::Dense<double> &C);
void adds_row(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void adds_row(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_row(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_row(const matrix::Dense<float> &A, const vector<float> &x,
matrix::Dense<float> &C);
void adds_row(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void adds_row(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void adds_row(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void adds_row(const matrix::Dense<double> &A, const size_t num,
const vector<double> &x, matrix::Dense<double> &C);
void adds_row(const matrix::Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void adds_row(const matrix::Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_row(const matrix::Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_row(const matrix::Dense<float> &A, const size_t num,
const vector<float> &x, matrix::Dense<float> &C);
void adds_row(const matrix::Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void adds_row(const matrix::Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void adds_row(const matrix::Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void adds_col(const matrix::Dense<double> &A, const vector<double> &x,
matrix::Dense<double> &C);
void adds_col(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void adds_col(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_col(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_col(const matrix::Dense<float> &A, const vector<float> &x,
matrix::Dense<float> &C);
void adds_col(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void adds_col(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void adds_col(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void adds_col(const matrix::Dense<double> &A, const size_t num,
const vector<double> &x, matrix::Dense<double> &C);
void adds_col(const matrix::Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
matrix::Dense<double> &C);
void adds_col(const matrix::Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_col(const matrix::Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
matrix::Dense<double> &C);
void adds_col(const matrix::Dense<float> &A, const size_t num,
const vector<float> &x, matrix::Dense<float> &C);
void adds_col(const matrix::Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x, matrix::Dense<float> &C);
void adds_col(const matrix::Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
matrix::Dense<float> &C);
void adds_col(const matrix::Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
matrix::Dense<float> &C);




void matvec(const matrix::Dense<double> &A, const vector<double> &x,
vector<double> &y);
void matvec(const matrix::Dense<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::Dense<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::Dense<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x, vector<double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::Dense<float> &A, const vector<float> &x,
vector<float> &y);
void matvec(const matrix::Dense<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::Dense<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::Dense<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, vector<float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void matvec_N(const matrix::Dense<double> &A, const vector<double> &x,
vector<double> &y);
void matvec_N(const matrix::Dense<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
vector<double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::Dense<float> &A, const vector<float> &x,
vector<float> &y);
void matvec_N(const matrix::Dense<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, vector<float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void matvec_T(const matrix::Dense<double> &A, const vector<double> &x,
vector<double> &y);
void matvec_T(const matrix::Dense<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
vector<double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::Dense<float> &A, const vector<float> &x,
vector<float> &y);
void matvec_T(const matrix::Dense<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x, vector<float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void matvec(const matrix::CRS<double> &A, const vector<double> &x,
vector<double> &y);
void matvec(const matrix::CRS<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::CRS<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::CRS<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x, vector<double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::CRS<float> &A, const vector<float> &x,
vector<float> &y);
void matvec(const matrix::CRS<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::CRS<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::CRS<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::CRS<float> &A, const view1D<vector<float>, float> &x,
vector<float> &y);
void matvec(const matrix::CRS<float> &A, const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::CRS<float> &A, const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::CRS<float> &A, const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void matvec_N(const matrix::CRS<double> &A, const vector<double> &x,
vector<double> &y);
void matvec_N(const matrix::CRS<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
vector<double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_N(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_N(const matrix::CRS<float> &A, const vector<float> &x,
vector<float> &y);
void matvec_N(const matrix::CRS<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x, vector<float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_N(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void matvec_T(const matrix::CRS<double> &A, const vector<double> &x,
vector<double> &y);
void matvec_T(const matrix::CRS<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
vector<double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec_T(const matrix::CRS<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec_T(const matrix::CRS<float> &A, const vector<float> &x,
vector<float> &y);
void matvec_T(const matrix::CRS<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x, vector<float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec_T(const matrix::CRS<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void matvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
vector<double> &y);
void matvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x, vector<double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void matvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
vector<float> &y);
void matvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x, vector<float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void matvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void rmatvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
vector<double> &y);
void rmatvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
view1D<vector<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x, vector<double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x, vector<double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void rmatvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
vector<float> &y);
void rmatvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
view1D<vector<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x, vector<float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void rmatvec(const matrix::LinearOperator<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);


} 
} 
