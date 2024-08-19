#pragma once
#include "../common/monolish_common.hpp"

namespace monolish {

namespace blas {





void copy(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void copy(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void tscal(const double alpha, tensor::tensor_Dense<double> &A);
void tscal(const float alpha, tensor::tensor_Dense<float> &A);




void times(const double alpha, const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void times(const float alpha, const tensor::tensor_Dense<float> &A,
tensor::tensor_Dense<float> &C);




void adds(const double alpha, const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void adds(const float alpha, const tensor::tensor_Dense<float> &A,
tensor::tensor_Dense<float> &C);




void tensadd(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void tensadd(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B,
tensor::tensor_Dense<float> &C);




void tenssub(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void tenssub(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B,
tensor::tensor_Dense<float> &C);




void tensmul(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void tensmul(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B,
tensor::tensor_Dense<float> &C);




void tensmul(const double &a, const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B, const double &b,
tensor::tensor_Dense<double> &C);
void tensmul(const float &a, const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, const float &b,
tensor::tensor_Dense<float> &C);

} 
} 
