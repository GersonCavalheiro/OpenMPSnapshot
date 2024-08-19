#pragma once

#include "../common/monolish_common.hpp"

namespace monolish {

namespace vml {





void add(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void add(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void sub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void sub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void mul(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void mul(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void div(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void div(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void add(const matrix::CRS<double> &A, const double alpha,
matrix::CRS<double> &C);
void add(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);




void sub(const matrix::CRS<double> &A, const double alpha,
matrix::CRS<double> &C);
void sub(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);




void mul(const matrix::CRS<double> &A, const double alpha,
matrix::CRS<double> &C);
void mul(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);




void div(const matrix::CRS<double> &A, const double alpha,
matrix::CRS<double> &C);
void div(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);




void pow(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void pow(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void pow(const matrix::CRS<double> &A, const double alpha,
matrix::CRS<double> &C);
void pow(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);




void sin(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void sin(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void sqrt(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void sqrt(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void sinh(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void sinh(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void asin(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void asin(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void asinh(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void asinh(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void tan(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void tan(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void tanh(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void tanh(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void atan(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void atan(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void atanh(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void atanh(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void ceil(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void ceil(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void floor(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void floor(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void sign(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void sign(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void max(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void max(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void min(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void min(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void max(const matrix::CRS<double> &A, const double alpha,
matrix::CRS<double> &C);
void max(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);




void min(const matrix::CRS<double> &A, const double alpha,
matrix::CRS<double> &C);
void min(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);




[[nodiscard]] double max(const matrix::CRS<double> &C);
[[nodiscard]] float max(const matrix::CRS<float> &C);




[[nodiscard]] double min(const matrix::CRS<double> &C);
[[nodiscard]] float min(const matrix::CRS<float> &C);




void alo(const matrix::CRS<double> &A, const double alpha, const double beta,
matrix::CRS<double> &C);
void alo(const matrix::CRS<float> &A, const float alpha, const float beta,
matrix::CRS<float> &C);




void reciprocal(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void reciprocal(const matrix::CRS<float> &A, matrix::CRS<float> &C);

} 
} 
