#pragma once

#include "../common/monolish_common.hpp"

namespace monolish {

namespace vml {





void add(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void add(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void sub(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void sub(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void mul(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void mul(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void div(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void div(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void add(const matrix::Dense<double> &A, const double alpha,
matrix::Dense<double> &C);
void add(const matrix::Dense<float> &A, const float alpha,
matrix::Dense<float> &C);




void sub(const matrix::Dense<double> &A, const double alpha,
matrix::Dense<double> &C);
void sub(const matrix::Dense<float> &A, const float alpha,
matrix::Dense<float> &C);




void mul(const matrix::Dense<double> &A, const double alpha,
matrix::Dense<double> &C);
void mul(const matrix::Dense<float> &A, const float alpha,
matrix::Dense<float> &C);




void div(const matrix::Dense<double> &A, const double alpha,
matrix::Dense<double> &C);
void div(const matrix::Dense<float> &A, const float alpha,
matrix::Dense<float> &C);




void pow(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void pow(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void pow(const matrix::Dense<double> &A, const double alpha,
matrix::Dense<double> &C);
void pow(const matrix::Dense<float> &A, const float alpha,
matrix::Dense<float> &C);




void sin(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void sin(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void sqrt(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void sqrt(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void sinh(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void sinh(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void asin(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void asin(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void asinh(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void asinh(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void tan(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void tan(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void tanh(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void tanh(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void atan(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void atan(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void atanh(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void atanh(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void ceil(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void ceil(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void floor(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void floor(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void sign(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void sign(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void exp(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void exp(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void max(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void max(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void min(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void min(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void max(const matrix::Dense<double> &A, const double alpha,
matrix::Dense<double> &C);
void max(const matrix::Dense<float> &A, const float alpha,
matrix::Dense<float> &C);




void min(const matrix::Dense<double> &A, const double alpha,
matrix::Dense<double> &C);
void min(const matrix::Dense<float> &A, const float alpha,
matrix::Dense<float> &C);




[[nodiscard]] double max(const matrix::Dense<double> &C);
[[nodiscard]] float max(const matrix::Dense<float> &C);




[[nodiscard]] double min(const matrix::Dense<double> &C);
[[nodiscard]] float min(const matrix::Dense<float> &C);




void alo(const matrix::Dense<double> &A, const double alpha, const double beta,
matrix::Dense<double> &C);
void alo(const matrix::Dense<float> &A, const float alpha, const float beta,
matrix::Dense<float> &C);




void reciprocal(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void reciprocal(const matrix::Dense<float> &A, matrix::Dense<float> &C);


} 
} 
