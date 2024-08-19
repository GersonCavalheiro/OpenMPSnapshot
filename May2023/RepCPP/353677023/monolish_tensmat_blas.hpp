#pragma once
#include "../common/monolish_common.hpp"

namespace monolish {

namespace blas {





void tensmat(const tensor::tensor_Dense<double> &A,
const matrix::Dense<double> &x, tensor::tensor_Dense<double> &y);
void tensmat(const tensor::tensor_Dense<float> &A,
const matrix::Dense<float> &x, tensor::tensor_Dense<float> &y);




void tensmat(const double &a, const tensor::tensor_Dense<double> &A,
const matrix::Dense<double> &x, const double &b,
tensor::tensor_Dense<double> &y);
void tensmat(const float &a, const tensor::tensor_Dense<float> &A,
const matrix::Dense<float> &x, const float &b,
tensor::tensor_Dense<float> &y);



} 
} 
