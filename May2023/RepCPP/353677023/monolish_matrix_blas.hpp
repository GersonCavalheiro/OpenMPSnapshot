#pragma once
#include "../common/monolish_common.hpp"

namespace monolish {

namespace blas {





void copy(const matrix::Dense<double> &A, matrix::Dense<double> &C);
void copy(const matrix::Dense<float> &A, matrix::Dense<float> &C);




void copy(const matrix::CRS<double> &A, matrix::CRS<double> &C);
void copy(const matrix::CRS<float> &A, matrix::CRS<float> &C);




void copy(const matrix::LinearOperator<double> &A,
matrix::LinearOperator<double> &C);
void copy(const matrix::LinearOperator<float> &A,
matrix::LinearOperator<float> &C);




void mscal(const double alpha, matrix::Dense<double> &A);
void mscal(const float alpha, matrix::Dense<float> &A);




void mscal(const double alpha, matrix::CRS<double> &A);
void mscal(const float alpha, matrix::CRS<float> &A);




void times(const double alpha, const matrix::Dense<double> &A,
matrix::Dense<double> &C);
void times(const float alpha, const matrix::Dense<float> &A,
matrix::Dense<float> &C);




void times(const double alpha, const matrix::CRS<double> &A,
matrix::CRS<double> &C);
void times(const float alpha, const matrix::CRS<float> &A,
matrix::CRS<float> &C);




void adds(const double alpha, const matrix::Dense<double> &A,
matrix::Dense<double> &C);
void adds(const float alpha, const matrix::Dense<float> &A,
matrix::Dense<float> &C);




void matadd(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void matadd(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void matadd(const matrix::LinearOperator<double> &A,
const matrix::LinearOperator<double> &B,
matrix::LinearOperator<double> &C);
void matadd(const matrix::LinearOperator<float> &A,
const matrix::LinearOperator<float> &B,
matrix::LinearOperator<float> &C);




void matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void matsub(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void matsub(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void matsub(const matrix::LinearOperator<double> &A,
const matrix::LinearOperator<double> &B,
matrix::LinearOperator<double> &C);
void matsub(const matrix::LinearOperator<float> &A,
const matrix::LinearOperator<float> &B,
matrix::LinearOperator<float> &C);




void matsub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
matrix::CRS<double> &C);
void matsub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
matrix::CRS<float> &C);




void matmul(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void matmul(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void matmul(const double &a, const matrix::Dense<double> &A,
const matrix::Dense<double> &B, const double &b,
matrix::Dense<double> &C);
void matmul(const float &a, const matrix::Dense<float> &A,
const matrix::Dense<float> &B, const float &b,
matrix::Dense<float> &C);




void matmul(const matrix::CRS<double> &A, const matrix::Dense<double> &B,
matrix::Dense<double> &C);
void matmul(const matrix::CRS<float> &A, const matrix::Dense<float> &B,
matrix::Dense<float> &C);




void matmul(const double &a, const matrix::CRS<double> &A,
const matrix::Dense<double> &B, const double &b,
matrix::Dense<double> &C);
void matmul(const float &a, const matrix::CRS<float> &A,
const matrix::Dense<float> &B, const float &b,
matrix::Dense<float> &C);




void matmul(const matrix::LinearOperator<double> &A,
const matrix::LinearOperator<double> &B,
matrix::LinearOperator<double> &C);
void matmul(const matrix::LinearOperator<float> &A,
const matrix::LinearOperator<float> &B,
matrix::LinearOperator<float> &C);




void matmul(const matrix::LinearOperator<double> &A,
const matrix::Dense<double> &B, matrix::Dense<double> &C);
void matmul(const matrix::LinearOperator<float> &A,
const matrix::Dense<float> &B, matrix::Dense<float> &C);




void rmatmul(const matrix::LinearOperator<double> &A,
const matrix::Dense<double> &B, matrix::Dense<double> &C);
void rmatmul(const matrix::LinearOperator<float> &A,
const matrix::Dense<float> &B, matrix::Dense<float> &C);


} 
} 
