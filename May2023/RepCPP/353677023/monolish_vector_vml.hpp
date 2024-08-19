#pragma once

#include "../common/monolish_common.hpp"

namespace monolish {

namespace vml {





void add(const vector<double> &a, const vector<double> &b, vector<double> &y);
void add(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void add(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void add(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void add(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void add(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void add(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const vector<float> &a, const vector<float> &b, vector<float> &y);
void add(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void add(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void add(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void add(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void add(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void add(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void sub(const vector<double> &a, const vector<double> &b, vector<double> &y);
void sub(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void sub(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void sub(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void sub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void sub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void sub(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const vector<float> &a, const vector<float> &b, vector<float> &y);
void sub(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void sub(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void sub(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void sub(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void sub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void sub(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void mul(const vector<double> &a, const vector<double> &b, vector<double> &y);
void mul(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void mul(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void mul(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void mul(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void mul(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void mul(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const vector<float> &a, const vector<float> &b, vector<float> &y);
void mul(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void mul(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void mul(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void mul(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void mul(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void mul(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void div(const vector<double> &a, const vector<double> &b, vector<double> &y);
void div(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void div(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void div(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void div(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void div(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void div(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const vector<float> &a, const vector<float> &b, vector<float> &y);
void div(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void div(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void div(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void div(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void div(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void div(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void add(const vector<double> &a, const double alpha, vector<double> &y);
void add(const vector<double> &a, const double alpha,
view1D<vector<double>, double> &y);
void add(const vector<double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void add(const vector<double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a, const double alpha,
vector<double> &y);
void add(const view1D<vector<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void add(const view1D<vector<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<vector<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a, const double alpha,
vector<double> &y);
void add(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void add(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, vector<double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<vector<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<matrix::Dense<double>, double> &y);
void add(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<tensor::tensor_Dense<double>, double> &y);
void add(const vector<float> &a, const float alpha, vector<float> &y);
void add(const vector<float> &a, const float alpha,
view1D<vector<float>, float> &y);
void add(const vector<float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void add(const vector<float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a, const float alpha,
vector<float> &y);
void add(const view1D<vector<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void add(const view1D<vector<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<vector<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const float alpha,
vector<float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
vector<float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void add(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);




void sub(const vector<double> &a, const double alpha, vector<double> &y);
void sub(const vector<double> &a, const double alpha,
view1D<vector<double>, double> &y);
void sub(const vector<double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void sub(const vector<double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a, const double alpha,
vector<double> &y);
void sub(const view1D<vector<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void sub(const view1D<vector<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<vector<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a, const double alpha,
vector<double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, vector<double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<vector<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<matrix::Dense<double>, double> &y);
void sub(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<tensor::tensor_Dense<double>, double> &y);
void sub(const vector<float> &a, const float alpha, vector<float> &y);
void sub(const vector<float> &a, const float alpha,
view1D<vector<float>, float> &y);
void sub(const vector<float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void sub(const vector<float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a, const float alpha,
vector<float> &y);
void sub(const view1D<vector<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void sub(const view1D<vector<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<vector<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const float alpha,
vector<float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
vector<float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void sub(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);




void mul(const vector<double> &a, const double alpha, vector<double> &y);
void mul(const vector<double> &a, const double alpha,
view1D<vector<double>, double> &y);
void mul(const vector<double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void mul(const vector<double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a, const double alpha,
vector<double> &y);
void mul(const view1D<vector<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void mul(const view1D<vector<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<vector<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a, const double alpha,
vector<double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, vector<double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<vector<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<matrix::Dense<double>, double> &y);
void mul(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<tensor::tensor_Dense<double>, double> &y);
void mul(const vector<float> &a, const float alpha, vector<float> &y);
void mul(const vector<float> &a, const float alpha,
view1D<vector<float>, float> &y);
void mul(const vector<float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void mul(const vector<float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a, const float alpha,
vector<float> &y);
void mul(const view1D<vector<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void mul(const view1D<vector<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<vector<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const float alpha,
vector<float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
vector<float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void mul(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);




void div(const vector<double> &a, const double alpha, vector<double> &y);
void div(const vector<double> &a, const double alpha,
view1D<vector<double>, double> &y);
void div(const vector<double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void div(const vector<double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a, const double alpha,
vector<double> &y);
void div(const view1D<vector<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void div(const view1D<vector<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<vector<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a, const double alpha,
vector<double> &y);
void div(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void div(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, vector<double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<vector<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<matrix::Dense<double>, double> &y);
void div(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<tensor::tensor_Dense<double>, double> &y);
void div(const vector<float> &a, const float alpha, vector<float> &y);
void div(const vector<float> &a, const float alpha,
view1D<vector<float>, float> &y);
void div(const vector<float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void div(const vector<float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a, const float alpha,
vector<float> &y);
void div(const view1D<vector<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void div(const view1D<vector<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<vector<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const float alpha,
vector<float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
vector<float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void div(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);




void pow(const vector<double> &a, const vector<double> &b, vector<double> &y);
void pow(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void pow(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void pow(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void pow(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void pow(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void pow(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const vector<float> &a, const vector<float> &b, vector<float> &y);
void pow(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void pow(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void pow(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void pow(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void pow(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void pow(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void pow(const vector<double> &a, const double alpha, vector<double> &y);
void pow(const vector<double> &a, const double alpha,
view1D<vector<double>, double> &y);
void pow(const vector<double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void pow(const vector<double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a, const double alpha,
vector<double> &y);
void pow(const view1D<vector<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void pow(const view1D<vector<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<vector<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a, const double alpha,
vector<double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, vector<double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<vector<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<matrix::Dense<double>, double> &y);
void pow(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<tensor::tensor_Dense<double>, double> &y);
void pow(const vector<float> &a, const float alpha, vector<float> &y);
void pow(const vector<float> &a, const float alpha,
view1D<vector<float>, float> &y);
void pow(const vector<float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void pow(const vector<float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a, const float alpha,
vector<float> &y);
void pow(const view1D<vector<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void pow(const view1D<vector<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<vector<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const float alpha,
vector<float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
vector<float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void pow(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);




void sin(const vector<double> &a, vector<double> &y);
void sin(const vector<double> &a, view1D<vector<double>, double> &y);
void sin(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void sin(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sin(const view1D<vector<double>, double> &a, vector<double> &y);
void sin(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void sin(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sin(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sin(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void sin(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sin(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sin(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sin(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void sin(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sin(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sin(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sin(const vector<float> &a, vector<float> &y);
void sin(const vector<float> &a, view1D<vector<float>, float> &y);
void sin(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void sin(const vector<float> &a, view1D<tensor::tensor_Dense<float>, float> &y);
void sin(const view1D<vector<float>, float> &a, vector<float> &y);
void sin(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void sin(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sin(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sin(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void sin(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sin(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sin(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sin(const view1D<tensor::tensor_Dense<float>, float> &a, vector<float> &y);
void sin(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sin(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sin(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void sqrt(const vector<double> &a, vector<double> &y);
void sqrt(const vector<double> &a, view1D<vector<double>, double> &y);
void sqrt(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void sqrt(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sqrt(const view1D<vector<double>, double> &a, vector<double> &y);
void sqrt(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void sqrt(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sqrt(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sqrt(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void sqrt(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sqrt(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sqrt(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sqrt(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void sqrt(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sqrt(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sqrt(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sqrt(const vector<float> &a, vector<float> &y);
void sqrt(const vector<float> &a, view1D<vector<float>, float> &y);
void sqrt(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void sqrt(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sqrt(const view1D<vector<float>, float> &a, vector<float> &y);
void sqrt(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void sqrt(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sqrt(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sqrt(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void sqrt(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sqrt(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sqrt(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sqrt(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void sqrt(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sqrt(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sqrt(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void sinh(const vector<double> &a, vector<double> &y);
void sinh(const vector<double> &a, view1D<vector<double>, double> &y);
void sinh(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void sinh(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sinh(const view1D<vector<double>, double> &a, vector<double> &y);
void sinh(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void sinh(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sinh(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sinh(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void sinh(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sinh(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sinh(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sinh(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void sinh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sinh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sinh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sinh(const vector<float> &a, vector<float> &y);
void sinh(const vector<float> &a, view1D<vector<float>, float> &y);
void sinh(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void sinh(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sinh(const view1D<vector<float>, float> &a, vector<float> &y);
void sinh(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void sinh(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sinh(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sinh(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void sinh(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sinh(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sinh(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sinh(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void sinh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sinh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sinh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void asin(const vector<double> &a, vector<double> &y);
void asin(const vector<double> &a, view1D<vector<double>, double> &y);
void asin(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void asin(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asin(const view1D<vector<double>, double> &a, vector<double> &y);
void asin(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void asin(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void asin(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asin(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void asin(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void asin(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void asin(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asin(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void asin(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void asin(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void asin(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asin(const vector<float> &a, vector<float> &y);
void asin(const vector<float> &a, view1D<vector<float>, float> &y);
void asin(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void asin(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void asin(const view1D<vector<float>, float> &a, vector<float> &y);
void asin(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void asin(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void asin(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void asin(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void asin(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void asin(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void asin(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void asin(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void asin(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void asin(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void asin(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void asinh(const vector<double> &a, vector<double> &y);
void asinh(const vector<double> &a, view1D<vector<double>, double> &y);
void asinh(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void asinh(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asinh(const view1D<vector<double>, double> &a, vector<double> &y);
void asinh(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void asinh(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void asinh(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asinh(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void asinh(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void asinh(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void asinh(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asinh(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void asinh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void asinh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void asinh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void asinh(const vector<float> &a, vector<float> &y);
void asinh(const vector<float> &a, view1D<vector<float>, float> &y);
void asinh(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void asinh(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void asinh(const view1D<vector<float>, float> &a, vector<float> &y);
void asinh(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void asinh(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void asinh(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void asinh(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void asinh(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void asinh(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void asinh(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void asinh(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void asinh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void asinh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void asinh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void tan(const vector<double> &a, vector<double> &y);
void tan(const vector<double> &a, view1D<vector<double>, double> &y);
void tan(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void tan(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tan(const view1D<vector<double>, double> &a, vector<double> &y);
void tan(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void tan(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void tan(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tan(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void tan(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void tan(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void tan(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tan(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void tan(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void tan(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void tan(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tan(const vector<float> &a, vector<float> &y);
void tan(const vector<float> &a, view1D<vector<float>, float> &y);
void tan(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void tan(const vector<float> &a, view1D<tensor::tensor_Dense<float>, float> &y);
void tan(const view1D<vector<float>, float> &a, vector<float> &y);
void tan(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void tan(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void tan(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void tan(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void tan(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void tan(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void tan(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void tan(const view1D<tensor::tensor_Dense<float>, float> &a, vector<float> &y);
void tan(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void tan(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void tan(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void tanh(const vector<double> &a, vector<double> &y);
void tanh(const vector<double> &a, view1D<vector<double>, double> &y);
void tanh(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void tanh(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tanh(const view1D<vector<double>, double> &a, vector<double> &y);
void tanh(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void tanh(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void tanh(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tanh(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void tanh(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void tanh(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void tanh(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tanh(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void tanh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void tanh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void tanh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void tanh(const vector<float> &a, vector<float> &y);
void tanh(const vector<float> &a, view1D<vector<float>, float> &y);
void tanh(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void tanh(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void tanh(const view1D<vector<float>, float> &a, vector<float> &y);
void tanh(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void tanh(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void tanh(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void tanh(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void tanh(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void tanh(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void tanh(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void tanh(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void tanh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void tanh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void tanh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void atan(const vector<double> &a, vector<double> &y);
void atan(const vector<double> &a, view1D<vector<double>, double> &y);
void atan(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void atan(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atan(const view1D<vector<double>, double> &a, vector<double> &y);
void atan(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void atan(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void atan(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atan(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void atan(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void atan(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void atan(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atan(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void atan(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void atan(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void atan(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atan(const vector<float> &a, vector<float> &y);
void atan(const vector<float> &a, view1D<vector<float>, float> &y);
void atan(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void atan(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void atan(const view1D<vector<float>, float> &a, vector<float> &y);
void atan(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void atan(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void atan(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void atan(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void atan(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void atan(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void atan(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void atan(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void atan(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void atan(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void atan(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void atanh(const vector<double> &a, vector<double> &y);
void atanh(const vector<double> &a, view1D<vector<double>, double> &y);
void atanh(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void atanh(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atanh(const view1D<vector<double>, double> &a, vector<double> &y);
void atanh(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void atanh(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void atanh(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atanh(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void atanh(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void atanh(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void atanh(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atanh(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void atanh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void atanh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void atanh(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void atanh(const vector<float> &a, vector<float> &y);
void atanh(const vector<float> &a, view1D<vector<float>, float> &y);
void atanh(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void atanh(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void atanh(const view1D<vector<float>, float> &a, vector<float> &y);
void atanh(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void atanh(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void atanh(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void atanh(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void atanh(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void atanh(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void atanh(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void atanh(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void atanh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void atanh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void atanh(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void ceil(const vector<double> &a, vector<double> &y);
void ceil(const vector<double> &a, view1D<vector<double>, double> &y);
void ceil(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void ceil(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void ceil(const view1D<vector<double>, double> &a, vector<double> &y);
void ceil(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void ceil(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void ceil(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void ceil(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void ceil(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void ceil(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void ceil(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void ceil(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void ceil(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void ceil(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void ceil(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void ceil(const vector<float> &a, vector<float> &y);
void ceil(const vector<float> &a, view1D<vector<float>, float> &y);
void ceil(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void ceil(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void ceil(const view1D<vector<float>, float> &a, vector<float> &y);
void ceil(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void ceil(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void ceil(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void ceil(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void ceil(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void ceil(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void ceil(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void ceil(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void ceil(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void ceil(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void ceil(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void floor(const vector<double> &a, vector<double> &y);
void floor(const vector<double> &a, view1D<vector<double>, double> &y);
void floor(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void floor(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void floor(const view1D<vector<double>, double> &a, vector<double> &y);
void floor(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void floor(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void floor(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void floor(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void floor(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void floor(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void floor(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void floor(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void floor(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void floor(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void floor(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void floor(const vector<float> &a, vector<float> &y);
void floor(const vector<float> &a, view1D<vector<float>, float> &y);
void floor(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void floor(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void floor(const view1D<vector<float>, float> &a, vector<float> &y);
void floor(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void floor(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void floor(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void floor(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void floor(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void floor(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void floor(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void floor(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void floor(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void floor(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void floor(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void sign(const vector<double> &a, vector<double> &y);
void sign(const vector<double> &a, view1D<vector<double>, double> &y);
void sign(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void sign(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sign(const view1D<vector<double>, double> &a, vector<double> &y);
void sign(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void sign(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sign(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sign(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void sign(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sign(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sign(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sign(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void sign(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void sign(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void sign(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void sign(const vector<float> &a, vector<float> &y);
void sign(const vector<float> &a, view1D<vector<float>, float> &y);
void sign(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void sign(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sign(const view1D<vector<float>, float> &a, vector<float> &y);
void sign(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void sign(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sign(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sign(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void sign(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sign(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sign(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void sign(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void sign(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void sign(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void sign(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void exp(const vector<double> &a, vector<double> &y);
void exp(const vector<double> &a, view1D<vector<double>, double> &y);
void exp(const vector<double> &a, view1D<matrix::Dense<double>, double> &y);
void exp(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void exp(const view1D<vector<double>, double> &a, vector<double> &y);
void exp(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void exp(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void exp(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void exp(const view1D<matrix::Dense<double>, double> &a, vector<double> &y);
void exp(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void exp(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void exp(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void exp(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void exp(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void exp(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void exp(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void exp(const vector<float> &a, vector<float> &y);
void exp(const vector<float> &a, view1D<vector<float>, float> &y);
void exp(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void exp(const vector<float> &a, view1D<tensor::tensor_Dense<float>, float> &y);
void exp(const view1D<vector<float>, float> &a, vector<float> &y);
void exp(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void exp(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void exp(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void exp(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void exp(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void exp(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void exp(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void exp(const view1D<tensor::tensor_Dense<float>, float> &a, vector<float> &y);
void exp(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void exp(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void exp(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);




void max(const vector<double> &a, const vector<double> &b, vector<double> &y);
void max(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void max(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void max(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void max(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void max(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void max(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const vector<float> &a, const vector<float> &b, vector<float> &y);
void max(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void max(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void max(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void max(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void max(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void max(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void min(const vector<double> &a, const vector<double> &b, vector<double> &y);
void min(const vector<double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void min(const vector<double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const vector<double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const vector<double> &a, const view1D<vector<double>, double> &b,
vector<double> &y);
void min(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const vector<double> &a, const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void min(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const vector<double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void min(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const vector<double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a, const vector<double> &b,
vector<double> &y);
void min(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a, const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, vector<double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<vector<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b, view1D<matrix::Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const vector<double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b, vector<double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<vector<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b, vector<double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<matrix::Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
vector<double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<vector<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const view1D<tensor::tensor_Dense<double>, double> &b,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const vector<float> &a, const vector<float> &b, vector<float> &y);
void min(const vector<float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void min(const vector<float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const vector<float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const vector<float> &a, const view1D<vector<float>, float> &b,
vector<float> &y);
void min(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const vector<float> &a, const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
vector<float> &y);
void min(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const vector<float> &a, const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void min(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const vector<float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a, const vector<float> &b,
vector<float> &y);
void min(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
vector<float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const vector<float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, vector<float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<vector<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<matrix::Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const vector<float> &b, view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b, vector<float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<vector<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b, vector<float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<matrix::Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b, vector<float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<vector<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a,
const view1D<tensor::tensor_Dense<float>, float> &b,
view1D<tensor::tensor_Dense<float>, float> &y);




void max(const vector<double> &a, const double alpha, vector<double> &y);
void max(const vector<double> &a, const double alpha,
view1D<vector<double>, double> &y);
void max(const vector<double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void max(const vector<double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a, const double alpha,
vector<double> &y);
void max(const view1D<vector<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void max(const view1D<vector<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<vector<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a, const double alpha,
vector<double> &y);
void max(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void max(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, vector<double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<vector<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<matrix::Dense<double>, double> &y);
void max(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<tensor::tensor_Dense<double>, double> &y);
void max(const vector<float> &a, const float alpha, vector<float> &y);
void max(const vector<float> &a, const float alpha,
view1D<vector<float>, float> &y);
void max(const vector<float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void max(const vector<float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a, const float alpha,
vector<float> &y);
void max(const view1D<vector<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void max(const view1D<vector<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<vector<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const float alpha,
vector<float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
vector<float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void max(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);




void min(const vector<double> &a, const double alpha, vector<double> &y);
void min(const vector<double> &a, const double alpha,
view1D<vector<double>, double> &y);
void min(const vector<double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void min(const vector<double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a, const double alpha,
vector<double> &y);
void min(const view1D<vector<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void min(const view1D<vector<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<vector<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a, const double alpha,
vector<double> &y);
void min(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<vector<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<matrix::Dense<double>, double> &y);
void min(const view1D<matrix::Dense<double>, double> &a, const double alpha,
view1D<tensor::tensor_Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, vector<double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<vector<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<matrix::Dense<double>, double> &y);
void min(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, view1D<tensor::tensor_Dense<double>, double> &y);
void min(const vector<float> &a, const float alpha, vector<float> &y);
void min(const vector<float> &a, const float alpha,
view1D<vector<float>, float> &y);
void min(const vector<float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void min(const vector<float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a, const float alpha,
vector<float> &y);
void min(const view1D<vector<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void min(const view1D<vector<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<vector<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const float alpha,
vector<float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<matrix::Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
vector<float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<vector<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<matrix::Dense<float>, float> &y);
void min(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
view1D<tensor::tensor_Dense<float>, float> &y);




[[nodiscard]] double max(const vector<double> &y);
[[nodiscard]] double max(const view1D<vector<double>, double> &y);
[[nodiscard]] double max(const view1D<matrix::Dense<double>, double> &y);
[[nodiscard]] double max(const view1D<tensor::tensor_Dense<double>, double> &y);
[[nodiscard]] float max(const vector<float> &y);
[[nodiscard]] float max(const view1D<vector<float>, float> &y);
[[nodiscard]] float max(const view1D<matrix::Dense<float>, float> &y);
[[nodiscard]] float max(const view1D<tensor::tensor_Dense<float>, float> &y);




[[nodiscard]] double min(const vector<double> &y);
[[nodiscard]] double min(const view1D<vector<double>, double> &y);
[[nodiscard]] double min(const view1D<matrix::Dense<double>, double> &y);
[[nodiscard]] double min(const view1D<tensor::tensor_Dense<double>, double> &y);
[[nodiscard]] float min(const vector<float> &y);
[[nodiscard]] float min(const view1D<vector<float>, float> &y);
[[nodiscard]] float min(const view1D<matrix::Dense<float>, float> &y);
[[nodiscard]] float min(const view1D<tensor::tensor_Dense<float>, float> &y);




void alo(const vector<double> &a, const double alpha, const double beta,
vector<double> &y);
void alo(const vector<double> &a, const double alpha, const double beta,
view1D<vector<double>, double> &y);
void alo(const vector<double> &a, const double alpha, const double beta,
view1D<matrix::Dense<double>, double> &y);
void alo(const vector<double> &a, const double alpha, const double beta,
view1D<tensor::tensor_Dense<double>, double> &y);
void alo(const view1D<vector<double>, double> &a, const double alpha,
const double beta, vector<double> &y);
void alo(const view1D<vector<double>, double> &a, const double alpha,
const double beta, view1D<vector<double>, double> &y);
void alo(const view1D<vector<double>, double> &a, const double alpha,
const double beta, view1D<matrix::Dense<double>, double> &y);
void alo(const view1D<vector<double>, double> &a, const double alpha,
const double beta, view1D<tensor::tensor_Dense<double>, double> &y);
void alo(const view1D<matrix::Dense<double>, double> &a, const double alpha,
const double beta, vector<double> &y);
void alo(const view1D<matrix::Dense<double>, double> &a, const double alpha,
const double beta, view1D<vector<double>, double> &y);
void alo(const view1D<matrix::Dense<double>, double> &a, const double alpha,
const double beta, view1D<matrix::Dense<double>, double> &y);
void alo(const view1D<matrix::Dense<double>, double> &a, const double alpha,
const double beta, view1D<tensor::tensor_Dense<double>, double> &y);
void alo(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, const double beta, vector<double> &y);
void alo(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, const double beta,
view1D<vector<double>, double> &y);
void alo(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, const double beta,
view1D<matrix::Dense<double>, double> &y);
void alo(const view1D<tensor::tensor_Dense<double>, double> &a,
const double alpha, const double beta,
view1D<tensor::tensor_Dense<double>, double> &y);
void alo(const vector<float> &a, const float alpha, const float beta,
vector<float> &y);
void alo(const vector<float> &a, const float alpha, const float beta,
view1D<vector<float>, float> &y);
void alo(const vector<float> &a, const float alpha, const float beta,
view1D<matrix::Dense<float>, float> &y);
void alo(const vector<float> &a, const float alpha, const float beta,
view1D<tensor::tensor_Dense<float>, float> &y);
void alo(const view1D<vector<float>, float> &a, const float alpha,
const float beta, vector<float> &y);
void alo(const view1D<vector<float>, float> &a, const float alpha,
const float beta, view1D<vector<float>, float> &y);
void alo(const view1D<vector<float>, float> &a, const float alpha,
const float beta, view1D<matrix::Dense<float>, float> &y);
void alo(const view1D<vector<float>, float> &a, const float alpha,
const float beta, view1D<tensor::tensor_Dense<float>, float> &y);
void alo(const view1D<matrix::Dense<float>, float> &a, const float alpha,
const float beta, vector<float> &y);
void alo(const view1D<matrix::Dense<float>, float> &a, const float alpha,
const float beta, view1D<vector<float>, float> &y);
void alo(const view1D<matrix::Dense<float>, float> &a, const float alpha,
const float beta, view1D<matrix::Dense<float>, float> &y);
void alo(const view1D<matrix::Dense<float>, float> &a, const float alpha,
const float beta, view1D<tensor::tensor_Dense<float>, float> &y);
void alo(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
const float beta, vector<float> &y);
void alo(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
const float beta, view1D<vector<float>, float> &y);
void alo(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
const float beta, view1D<matrix::Dense<float>, float> &y);
void alo(const view1D<tensor::tensor_Dense<float>, float> &a, const float alpha,
const float beta, view1D<tensor::tensor_Dense<float>, float> &y);




void reciprocal(const vector<double> &a, vector<double> &y);
void reciprocal(const vector<double> &a, view1D<vector<double>, double> &y);
void reciprocal(const vector<double> &a,
view1D<matrix::Dense<double>, double> &y);
void reciprocal(const vector<double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void reciprocal(const view1D<vector<double>, double> &a, vector<double> &y);
void reciprocal(const view1D<vector<double>, double> &a,
view1D<vector<double>, double> &y);
void reciprocal(const view1D<vector<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void reciprocal(const view1D<vector<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void reciprocal(const view1D<matrix::Dense<double>, double> &a,
vector<double> &y);
void reciprocal(const view1D<matrix::Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void reciprocal(const view1D<matrix::Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void reciprocal(const view1D<matrix::Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void reciprocal(const view1D<tensor::tensor_Dense<double>, double> &a,
vector<double> &y);
void reciprocal(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<vector<double>, double> &y);
void reciprocal(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<matrix::Dense<double>, double> &y);
void reciprocal(const view1D<tensor::tensor_Dense<double>, double> &a,
view1D<tensor::tensor_Dense<double>, double> &y);
void reciprocal(const vector<float> &a, vector<float> &y);
void reciprocal(const vector<float> &a, view1D<vector<float>, float> &y);
void reciprocal(const vector<float> &a, view1D<matrix::Dense<float>, float> &y);
void reciprocal(const vector<float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void reciprocal(const view1D<vector<float>, float> &a, vector<float> &y);
void reciprocal(const view1D<vector<float>, float> &a,
view1D<vector<float>, float> &y);
void reciprocal(const view1D<vector<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void reciprocal(const view1D<vector<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void reciprocal(const view1D<matrix::Dense<float>, float> &a, vector<float> &y);
void reciprocal(const view1D<matrix::Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void reciprocal(const view1D<matrix::Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void reciprocal(const view1D<matrix::Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);
void reciprocal(const view1D<tensor::tensor_Dense<float>, float> &a,
vector<float> &y);
void reciprocal(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<vector<float>, float> &y);
void reciprocal(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<matrix::Dense<float>, float> &y);
void reciprocal(const view1D<tensor::tensor_Dense<float>, float> &a,
view1D<tensor::tensor_Dense<float>, float> &y);


} 
} 
