#pragma once
#include "../common/monolish_common.hpp"

namespace monolish {

namespace blas {





void times_row(const tensor::tensor_Dense<double> &A, const vector<double> &x,
tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<double> &A,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<float> &A, const vector<float> &x,
tensor::tensor_Dense<float> &C);
void times_row(const tensor::tensor_Dense<float> &A,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_row(const tensor::tensor_Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_row(const tensor::tensor_Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void times_row(const tensor::tensor_Dense<double> &A, const size_t num,
const vector<double> &x, tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_row(const tensor::tensor_Dense<float> &A, const size_t num,
const vector<float> &x, tensor::tensor_Dense<float> &C);
void times_row(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_row(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_row(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void times_col(const tensor::tensor_Dense<double> &A, const vector<double> &x,
tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<double> &A,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<float> &A, const vector<float> &x,
tensor::tensor_Dense<float> &C);
void times_col(const tensor::tensor_Dense<float> &A,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_col(const tensor::tensor_Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_col(const tensor::tensor_Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void times_col(const tensor::tensor_Dense<double> &A, const size_t num,
const vector<double> &x, tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void times_col(const tensor::tensor_Dense<float> &A, const size_t num,
const vector<float> &x, tensor::tensor_Dense<float> &C);
void times_col(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_col(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void times_col(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void adds_row(const tensor::tensor_Dense<double> &A, const vector<double> &x,
tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<double> &A,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<float> &A, const vector<float> &x,
tensor::tensor_Dense<float> &C);
void adds_row(const tensor::tensor_Dense<float> &A,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_row(const tensor::tensor_Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_row(const tensor::tensor_Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void adds_row(const tensor::tensor_Dense<double> &A, const size_t num,
const vector<double> &x, tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_row(const tensor::tensor_Dense<float> &A, const size_t num,
const vector<float> &x, tensor::tensor_Dense<float> &C);
void adds_row(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_row(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_row(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void adds_col(const tensor::tensor_Dense<double> &A, const vector<double> &x,
tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<double> &A,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<float> &A, const vector<float> &x,
tensor::tensor_Dense<float> &C);
void adds_col(const tensor::tensor_Dense<float> &A,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_col(const tensor::tensor_Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_col(const tensor::tensor_Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void adds_col(const tensor::tensor_Dense<double> &A, const size_t num,
const vector<double> &x, tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<double> &A, const size_t num,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &C);
void adds_col(const tensor::tensor_Dense<float> &A, const size_t num,
const vector<float> &x, tensor::tensor_Dense<float> &C);
void adds_col(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_col(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);
void adds_col(const tensor::tensor_Dense<float> &A, const size_t num,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &C);




void tensvec(const tensor::tensor_Dense<double> &A, const vector<double> &x,
tensor::tensor_Dense<double> &y);
void tensvec(const tensor::tensor_Dense<double> &A,
const view1D<vector<double>, double> &x,
tensor::tensor_Dense<double> &y);
void tensvec(const tensor::tensor_Dense<double> &A,
const view1D<matrix::Dense<double>, double> &x,
tensor::tensor_Dense<double> &y);
void tensvec(const tensor::tensor_Dense<double> &A,
const view1D<tensor::tensor_Dense<double>, double> &x,
tensor::tensor_Dense<double> &y);
void tensvec(const tensor::tensor_Dense<float> &A, const vector<float> &x,
tensor::tensor_Dense<float> &y);
void tensvec(const tensor::tensor_Dense<float> &A,
const view1D<vector<float>, float> &x,
tensor::tensor_Dense<float> &y);
void tensvec(const tensor::tensor_Dense<float> &A,
const view1D<matrix::Dense<float>, float> &x,
tensor::tensor_Dense<float> &y);
void tensvec(const tensor::tensor_Dense<float> &A,
const view1D<tensor::tensor_Dense<float>, float> &x,
tensor::tensor_Dense<float> &y);



} 
} 
