#pragma once

#include "../common/monolish_common.hpp"

namespace monolish {

namespace vml {





void add(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void add(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, tensor::tensor_Dense<float> &C);




void sub(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void sub(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, tensor::tensor_Dense<float> &C);




void mul(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void mul(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, tensor::tensor_Dense<float> &C);




void div(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void div(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, tensor::tensor_Dense<float> &C);




void add(const tensor::tensor_Dense<double> &A, const double alpha,
tensor::tensor_Dense<double> &C);
void add(const tensor::tensor_Dense<float> &A, const float alpha,
tensor::tensor_Dense<float> &C);




void sub(const tensor::tensor_Dense<double> &A, const double alpha,
tensor::tensor_Dense<double> &C);
void sub(const tensor::tensor_Dense<float> &A, const float alpha,
tensor::tensor_Dense<float> &C);




void mul(const tensor::tensor_Dense<double> &A, const double alpha,
tensor::tensor_Dense<double> &C);
void mul(const tensor::tensor_Dense<float> &A, const float alpha,
tensor::tensor_Dense<float> &C);




void div(const tensor::tensor_Dense<double> &A, const double alpha,
tensor::tensor_Dense<double> &C);
void div(const tensor::tensor_Dense<float> &A, const float alpha,
tensor::tensor_Dense<float> &C);




void pow(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void pow(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, tensor::tensor_Dense<float> &C);




void pow(const tensor::tensor_Dense<double> &A, const double alpha,
tensor::tensor_Dense<double> &C);
void pow(const tensor::tensor_Dense<float> &A, const float alpha,
tensor::tensor_Dense<float> &C);




void sin(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void sin(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void sqrt(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void sqrt(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void sinh(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void sinh(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void asin(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void asin(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void asinh(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void asinh(const tensor::tensor_Dense<float> &A,
tensor::tensor_Dense<float> &C);




void tan(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void tan(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void tanh(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void tanh(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void atan(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void atan(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void atanh(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void atanh(const tensor::tensor_Dense<float> &A,
tensor::tensor_Dense<float> &C);




void ceil(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void ceil(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void floor(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void floor(const tensor::tensor_Dense<float> &A,
tensor::tensor_Dense<float> &C);




void sign(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void sign(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void exp(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void exp(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C);




void max(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void max(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, tensor::tensor_Dense<float> &C);




void min(const tensor::tensor_Dense<double> &A,
const tensor::tensor_Dense<double> &B,
tensor::tensor_Dense<double> &C);
void min(const tensor::tensor_Dense<float> &A,
const tensor::tensor_Dense<float> &B, tensor::tensor_Dense<float> &C);




void max(const tensor::tensor_Dense<double> &A, const double alpha,
tensor::tensor_Dense<double> &C);
void max(const tensor::tensor_Dense<float> &A, const float alpha,
tensor::tensor_Dense<float> &C);




void min(const tensor::tensor_Dense<double> &A, const double alpha,
tensor::tensor_Dense<double> &C);
void min(const tensor::tensor_Dense<float> &A, const float alpha,
tensor::tensor_Dense<float> &C);




[[nodiscard]] double max(const tensor::tensor_Dense<double> &C);
[[nodiscard]] float max(const tensor::tensor_Dense<float> &C);




[[nodiscard]] double min(const tensor::tensor_Dense<double> &C);
[[nodiscard]] float min(const tensor::tensor_Dense<float> &C);




void alo(const tensor::tensor_Dense<double> &A, const double alpha,
const double beta, tensor::tensor_Dense<double> &C);
void alo(const tensor::tensor_Dense<float> &A, const float alpha,
const float beta, tensor::tensor_Dense<float> &C);




void reciprocal(const tensor::tensor_Dense<double> &A,
tensor::tensor_Dense<double> &C);
void reciprocal(const tensor::tensor_Dense<float> &A,
tensor::tensor_Dense<float> &C);


} 
} 
