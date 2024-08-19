#pragma once

#include "../common/monolish_common.hpp"

namespace monolish {

namespace vml {





void add(const matrix::LinearOperator<double> &A,
const matrix::LinearOperator<double> &B,
matrix::LinearOperator<double> &C);
void add(const matrix::LinearOperator<float> &A,
const matrix::LinearOperator<float> &B,
matrix::LinearOperator<float> &C);




void sub(const matrix::LinearOperator<double> &A,
const matrix::LinearOperator<double> &B,
matrix::LinearOperator<double> &C);
void sub(const matrix::LinearOperator<float> &A,
const matrix::LinearOperator<float> &B,
matrix::LinearOperator<float> &C);




void add(const matrix::LinearOperator<double> &A, const double &alpha,
matrix::LinearOperator<double> &C);
void add(const matrix::LinearOperator<float> &A, const float &alpha,
matrix::LinearOperator<float> &C);




void sub(const matrix::LinearOperator<double> &A, const double &alpha,
matrix::LinearOperator<double> &C);
void sub(const matrix::LinearOperator<float> &A, const float &alpha,
matrix::LinearOperator<float> &C);




void mul(const matrix::LinearOperator<double> &A, const double &alpha,
matrix::LinearOperator<double> &C);
void mul(const matrix::LinearOperator<float> &A, const float &alpha,
matrix::LinearOperator<float> &C);




void div(const matrix::LinearOperator<double> &A, const double &alpha,
matrix::LinearOperator<double> &C);
void div(const matrix::LinearOperator<float> &A, const float &alpha,
matrix::LinearOperator<float> &C);


} 
} 
