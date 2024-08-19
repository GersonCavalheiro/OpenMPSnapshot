#pragma once
#include "../common/monolish_common.hpp"

namespace monolish {

namespace blas {





void copy(const vector<double> &x, vector<double> &y);
void copy(const vector<double> &x, view1D<vector<double>, double> &y);
void copy(const vector<double> &x, view1D<matrix::Dense<double>, double> &y);
void copy(const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void copy(const view1D<vector<double>, double> &x, vector<double> &y);
void copy(const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void copy(const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void copy(const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void copy(const view1D<matrix::Dense<double>, double> &x, vector<double> &y);
void copy(const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void copy(const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void copy(const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void copy(const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void copy(const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void copy(const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void copy(const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void copy(const vector<float> &x, vector<float> &y);
void copy(const vector<float> &x, view1D<vector<float>, float> &y);
void copy(const vector<float> &x, view1D<matrix::Dense<float>, float> &y);
void copy(const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void copy(const view1D<vector<float>, float> &x, vector<float> &y);
void copy(const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void copy(const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void copy(const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void copy(const view1D<matrix::Dense<float>, float> &x, vector<float> &y);
void copy(const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void copy(const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void copy(const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void copy(const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void copy(const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void copy(const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void copy(const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void vecadd(const vector<double> &a, const vector<double> &b,
vector<double> &y);
void vecadd(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void vecadd(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void vecadd(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecadd(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecadd(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void vecadd(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecadd(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecadd(const vector<float> &a, const vector<float> &b, vector<float> &y);
void vecadd(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void vecadd(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void vecadd(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecadd(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecadd(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void vecadd(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecadd(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void vecsub(const vector<double> &a, const vector<double> &b,
vector<double> &y);
void vecsub(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void vecsub(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecsub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecsub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void vecsub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void vecsub(const vector<float> &a, const vector<float> &b, vector<float> &y);
void vecsub(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void vecsub(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecsub(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const vector<float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecsub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void vecsub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void times(const double alpha, const vector<double> &a, vector<double> &y);
void times(const double alpha, const vector<double> &a,
view1D<vector<double>, double> &y);
void times(const double alpha, const vector<double> &a,
view1D<matrix::Dense<double>, double> &y);
void times(const double alpha, const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const double alpha, const view1D<vector<double>, double> &a,
vector<double> &y);
void times(const double alpha, const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void times(const double alpha, const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void times(const double alpha, const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const double alpha, const view1D<matrix::Dense<double>, double> &a,
vector<double> &y);
void times(const double alpha, const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void times(const double alpha, const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void times(const double alpha, const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void times(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void times(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void times(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const float alpha, const vector<float> &a, vector<float> &y);
void times(const float alpha, const vector<float> &a,
view1D<vector<float>, float> &y);
void times(const float alpha, const vector<float> &a,
view1D<matrix::Dense<float>, float> &y);
void times(const float alpha, const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const float alpha, const view1D<vector<float>, float> &a,
vector<float> &y);
void times(const float alpha, const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void times(const float alpha, const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void times(const float alpha, const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const float alpha, const view1D<matrix::Dense<float>, float> &a,
vector<float> &y);
void times(const float alpha, const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void times(const float alpha, const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void times(const float alpha, const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void times(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void times(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void times(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void times(const vector<double> &a, const vector<double> &b, vector<double> &y);
void times(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void times(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void times(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void times(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void times(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void times(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void times(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void times(const vector<float> &a, const vector<float> &b, vector<float> &y);
void times(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void times(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void times(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void times(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void times(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void times(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void times(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
vector<float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void times(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void asum(const vector<double> &x, double &ans);
void asum(const view1D<vector<double>, double> &x, double &ans);
void asum(const view1D<matrix::Dense<double>, double> &x, double &ans);
void asum(const view1D<tensor::tensor_Dense<double>, double> &x, double &ans);
void asum(const vector<float> &x, float &ans);
void asum(const view1D<vector<float>, float> &x, float &ans);
void asum(const view1D<matrix::Dense<float>, float> &x, float &ans);
void asum(const view1D<tensor::tensor_Dense<float>, float> &x, float &ans);


[[nodiscard]] double asum(const vector<double> &x);
[[nodiscard]] double asum(const view1D<vector<double>, double> &x);
[[nodiscard]] double asum(const view1D<matrix::Dense<double>, double> &x);
[[nodiscard]] double
asum(const view1D<tensor::tensor_Dense<double>, double> &x);
[[nodiscard]] float asum(const vector<float> &x);
[[nodiscard]] float asum(const view1D<vector<float>, float> &x);
[[nodiscard]] float asum(const view1D<matrix::Dense<float>, float> &x);
[[nodiscard]] float asum(const view1D<tensor::tensor_Dense<float>, float> &x);




void sum(const vector<double> &x, double &ans);
void sum(const view1D<vector<double>, double> &x, double &ans);
void sum(const view1D<matrix::Dense<double>, double> &x, double &ans);
void sum(const view1D<tensor::tensor_Dense<double>, double> &x, double &ans);
void sum(const vector<float> &x, float &ans);
void sum(const view1D<vector<float>, float> &x, float &ans);
void sum(const view1D<matrix::Dense<float>, float> &x, float &ans);
void sum(const view1D<tensor::tensor_Dense<float>, float> &x, float &ans);


[[nodiscard]] double sum(const vector<double> &x);
[[nodiscard]] double sum(const view1D<vector<double>, double> &x);
[[nodiscard]] double sum(const view1D<matrix::Dense<double>, double> &x);
[[nodiscard]] double sum(const view1D<tensor::tensor_Dense<double>, double> &x);
[[nodiscard]] float sum(const vector<float> &x);
[[nodiscard]] float sum(const view1D<vector<float>, float> &x);
[[nodiscard]] float sum(const view1D<matrix::Dense<float>, float> &x);
[[nodiscard]] float sum(const view1D<tensor::tensor_Dense<float>, float> &x);




void axpy(const double alpha, const vector<double> &x, vector<double> &y);
void axpy(const double alpha, const vector<double> &x,
view1D<vector<double>, double> &y);
void axpy(const double alpha, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void axpy(const double alpha, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void axpy(const double alpha, const view1D<vector<double>, double> &x,
vector<double> &y);
void axpy(const double alpha, const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void axpy(const double alpha, const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void axpy(const double alpha, const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void axpy(const double alpha, const view1D<matrix::Dense<double>, double> &x,
vector<double> &y);
void axpy(const double alpha, const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void axpy(const double alpha, const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void axpy(const double alpha, const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void axpy(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void axpy(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void axpy(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void axpy(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void axpy(const float alpha, const vector<float> &x, vector<float> &y);
void axpy(const float alpha, const vector<float> &x,
view1D<vector<float>, float> &y);
void axpy(const float alpha, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void axpy(const float alpha, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void axpy(const float alpha, const view1D<vector<float>, float> &x,
vector<float> &y);
void axpy(const float alpha, const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void axpy(const float alpha, const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void axpy(const float alpha, const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void axpy(const float alpha, const view1D<matrix::Dense<float>, float> &x,
vector<float> &y);
void axpy(const float alpha, const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void axpy(const float alpha, const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void axpy(const float alpha, const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void axpy(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void axpy(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void axpy(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void axpy(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);




void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
vector<double> &z);
void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<vector<double>, double> &y, vector<double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<vector<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<vector<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<vector<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y, vector<double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
vector<double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const vector<double> &y, vector<double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const vector<double> &y, view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const vector<double> &y, view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const vector<double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y, vector<double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y, vector<double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
vector<double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y, vector<double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y, view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y, view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y, vector<double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y, vector<double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
vector<double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha, const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y, vector<double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y, view1D<vector<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y, view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y, vector<double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y, vector<double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
vector<double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<vector<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<matrix::Dense<double>, double> &z);
void axpyz(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y,
view1D<tensor::tensor_Dense<double>, double> &z);
void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
vector<float> &z);
void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<vector<float>, float> &y, vector<float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<vector<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<vector<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<vector<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<matrix::Dense<float>, float> &y, vector<float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
vector<float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const vector<float> &y, vector<float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const vector<float> &y, view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const vector<float> &y, view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const vector<float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y, vector<float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y, vector<float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
vector<float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const vector<float> &y, vector<float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const vector<float> &y, view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const vector<float> &y, view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const vector<float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y, vector<float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y, vector<float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
vector<float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha, const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y, vector<float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y, view1D<vector<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y, view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y, vector<float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y, vector<float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
vector<float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<vector<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<matrix::Dense<float>, float> &z);
void axpyz(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y,
view1D<tensor::tensor_Dense<float>, float> &z);




void dot(const vector<double> &x, const vector<double> &y, double &ans);
void dot(const vector<double> &x, const view1D<vector<double>, double> &y,
double &ans);
void dot(const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y, double &ans);
void dot(const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y, double &ans);
void dot(const view1D<vector<double>, double> &x, const vector<double> &y,
double &ans);
void dot(const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y, double &ans);
void dot(const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y, double &ans);
void dot(const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y, double &ans);
void dot(const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y, double &ans);
void dot(const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y, double &ans);
void dot(const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y, double &ans);
void dot(const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y, double &ans);
void dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y, double &ans);
void dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y, double &ans);
void dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y, double &ans);
void dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y, double &ans);
void dot(const vector<float> &x, const vector<float> &y, float &ans);
void dot(const vector<float> &x, const view1D<vector<float>, float> &y,
float &ans);
void dot(const vector<float> &x, const view1D<matrix::Dense<float>, float> &y,
float &ans);
void dot(const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y, float &ans);
void dot(const view1D<vector<float>, float> &x, const vector<float> &y,
float &ans);
void dot(const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y, float &ans);
void dot(const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y, float &ans);
void dot(const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y, float &ans);
void dot(const view1D<matrix::Dense<float>, float> &x, const vector<float> &y,
float &ans);
void dot(const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y, float &ans);
void dot(const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y, float &ans);
void dot(const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y, float &ans);
void dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y, float &ans);
void dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y, float &ans);
void dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y, float &ans);
void dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y, float &ans);


[[nodiscard]] double dot(const vector<double> &x, const vector<double> &y);
[[nodiscard]] double dot(const vector<double> &x,
const view1D<vector<double>, double> &y);
[[nodiscard]] double dot(const vector<double> &x,
const view1D<matrix::Dense<double>, double> &y);
[[nodiscard]] double dot(const vector<double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
[[nodiscard]] double dot(const view1D<vector<double>, double> &x,
const vector<double> &y);
[[nodiscard]] double dot(const view1D<vector<double>, double> &x,
const view1D<vector<double>, double> &y);
[[nodiscard]] double dot(const view1D<vector<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
[[nodiscard]] double dot(const view1D<vector<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
[[nodiscard]] double dot(const view1D<matrix::Dense<double>, double> &x,
const vector<double> &y);
[[nodiscard]] double dot(const view1D<matrix::Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
[[nodiscard]] double dot(const view1D<matrix::Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
[[nodiscard]] double dot(const view1D<matrix::Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
[[nodiscard]] double dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const vector<double> &y);
[[nodiscard]] double dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<vector<double>, double> &y);
[[nodiscard]] double dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<matrix::Dense<double>, double> &y);
[[nodiscard]] double dot(const view1D<tensor::tensor_Dense<double>, double> &x,
const view1D<tensor::tensor_Dense<double>, double> &y);
[[nodiscard]] float dot(const vector<float> &x, const vector<float> &y);
[[nodiscard]] float dot(const vector<float> &x,
const view1D<vector<float>, float> &y);
[[nodiscard]] float dot(const vector<float> &x,
const view1D<matrix::Dense<float>, float> &y);
[[nodiscard]] float dot(const vector<float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
[[nodiscard]] float dot(const view1D<vector<float>, float> &x,
const vector<float> &y);
[[nodiscard]] float dot(const view1D<vector<float>, float> &x,
const view1D<vector<float>, float> &y);
[[nodiscard]] float dot(const view1D<vector<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
[[nodiscard]] float dot(const view1D<vector<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
[[nodiscard]] float dot(const view1D<matrix::Dense<float>, float> &x,
const vector<float> &y);
[[nodiscard]] float dot(const view1D<matrix::Dense<float>, float> &x,
const view1D<vector<float>, float> &y);
[[nodiscard]] float dot(const view1D<matrix::Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
[[nodiscard]] float dot(const view1D<matrix::Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);
[[nodiscard]] float dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const vector<float> &y);
[[nodiscard]] float dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<vector<float>, float> &y);
[[nodiscard]] float dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<matrix::Dense<float>, float> &y);
[[nodiscard]] float dot(const view1D<tensor::tensor_Dense<float>, float> &x,
const view1D<tensor::tensor_Dense<float>, float> &y);




void nrm1(const vector<double> &x, double &ans);
void nrm1(const view1D<vector<double>, double> &x, double &ans);
void nrm1(const view1D<matrix::Dense<double>, double> &x, double &ans);
void nrm1(const view1D<tensor::tensor_Dense<double>, double> &x, double &ans);
void nrm1(const vector<float> &x, float &ans);
void nrm1(const view1D<vector<float>, float> &x, float &ans);
void nrm1(const view1D<matrix::Dense<float>, float> &x, float &ans);
void nrm1(const view1D<tensor::tensor_Dense<float>, float> &x, float &ans);


[[nodiscard]] double nrm1(const vector<double> &x);
[[nodiscard]] double nrm1(const view1D<vector<double>, double> &x);
[[nodiscard]] double nrm1(const view1D<matrix::Dense<double>, double> &x);
[[nodiscard]] double
nrm1(const view1D<tensor::tensor_Dense<double>, double> &x);
[[nodiscard]] float nrm1(const vector<float> &x);
[[nodiscard]] float nrm1(const view1D<vector<float>, float> &x);
[[nodiscard]] float nrm1(const view1D<matrix::Dense<float>, float> &x);
[[nodiscard]] float nrm1(const view1D<tensor::tensor_Dense<float>, float> &x);




void nrm2(const vector<double> &x, double &ans);
void nrm2(const view1D<vector<double>, double> &x, double &ans);
void nrm2(const view1D<matrix::Dense<double>, double> &x, double &ans);
void nrm2(const view1D<tensor::tensor_Dense<double>, double> &x, double &ans);
void nrm2(const vector<float> &x, float &ans);
void nrm2(const view1D<vector<float>, float> &x, float &ans);
void nrm2(const view1D<matrix::Dense<float>, float> &x, float &ans);
void nrm2(const view1D<tensor::tensor_Dense<float>, float> &x, float &ans);


[[nodiscard]] double nrm2(const vector<double> &x);
[[nodiscard]] double nrm2(const view1D<vector<double>, double> &x);
[[nodiscard]] double nrm2(const view1D<matrix::Dense<double>, double> &x);
[[nodiscard]] double
nrm2(const view1D<tensor::tensor_Dense<double>, double> &x);
[[nodiscard]] float nrm2(const vector<float> &x);
[[nodiscard]] float nrm2(const view1D<vector<float>, float> &x);
[[nodiscard]] float nrm2(const view1D<matrix::Dense<float>, float> &x);
[[nodiscard]] float nrm2(const view1D<tensor::tensor_Dense<float>, float> &x);




void scal(const double alpha, vector<double> &x);
void scal(const double alpha, view1D<vector<double>, double> &x);
void scal(const double alpha, view1D<matrix::Dense<double>, double> &x);
void scal(const double alpha, view1D<tensor::tensor_Dense<double>, double> &x);
void scal(const float alpha, vector<float> &x);
void scal(const float alpha, view1D<vector<float>, float> &x);
void scal(const float alpha, view1D<matrix::Dense<float>, float> &x);
void scal(const float alpha, view1D<tensor::tensor_Dense<float>, float> &x);




void xpay(const double alpha, const vector<double> &x, vector<double> &y);
void xpay(const double alpha, const vector<double> &x,
view1D<vector<double>, double> &y);
void xpay(const double alpha, const vector<double> &x,
view1D<matrix::Dense<double>, double> &y);
void xpay(const double alpha, const vector<double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void xpay(const double alpha, const view1D<vector<double>, double> &x,
vector<double> &y);
void xpay(const double alpha, const view1D<vector<double>, double> &x,
view1D<vector<double>, double> &y);
void xpay(const double alpha, const view1D<vector<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void xpay(const double alpha, const view1D<vector<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void xpay(const double alpha, const view1D<matrix::Dense<double>, double> &x,
vector<double> &y);
void xpay(const double alpha, const view1D<matrix::Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void xpay(const double alpha, const view1D<matrix::Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void xpay(const double alpha, const view1D<matrix::Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void xpay(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
vector<double> &y);
void xpay(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<vector<double>, double> &y);
void xpay(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<matrix::Dense<double>, double> &y);
void xpay(const double alpha,
const view1D<tensor::tensor_Dense<double>, double> &x,
view1D<tensor::tensor_Dense<double>, double> &y);
void xpay(const float alpha, const vector<float> &x, vector<float> &y);
void xpay(const float alpha, const vector<float> &x,
view1D<vector<float>, float> &y);
void xpay(const float alpha, const vector<float> &x,
view1D<matrix::Dense<float>, float> &y);
void xpay(const float alpha, const vector<float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void xpay(const float alpha, const view1D<vector<float>, float> &x,
vector<float> &y);
void xpay(const float alpha, const view1D<vector<float>, float> &x,
view1D<vector<float>, float> &y);
void xpay(const float alpha, const view1D<vector<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void xpay(const float alpha, const view1D<vector<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void xpay(const float alpha, const view1D<matrix::Dense<float>, float> &x,
vector<float> &y);
void xpay(const float alpha, const view1D<matrix::Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void xpay(const float alpha, const view1D<matrix::Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void xpay(const float alpha, const view1D<matrix::Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);
void xpay(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
vector<float> &y);
void xpay(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<vector<float>, float> &y);
void xpay(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<matrix::Dense<float>, float> &y);
void xpay(const float alpha,
const view1D<tensor::tensor_Dense<float>, float> &x,
view1D<tensor::tensor_Dense<float>, float> &y);


} 
} 
