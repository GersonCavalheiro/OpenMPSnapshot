#pragma once
#include "../common/monolish_common.hpp"

namespace monolish {

namespace blas {





void mattens(const matrix::Dense<double> &A,
const tensor::tensor_Dense<double> &x,
tensor::tensor_Dense<double> &y);
void mattens(const matrix::Dense<float> &A,
const tensor::tensor_Dense<float> &x,
tensor::tensor_Dense<float> &y);




void mattens(const double &a, const matrix::Dense<double> &A,
const tensor::tensor_Dense<double> &x, const double &b,
tensor::tensor_Dense<double> &y);
void mattens(const float &a, const matrix::Dense<float> &A,
const tensor::tensor_Dense<float> &x, const float &b,
tensor::tensor_Dense<float> &y);




void mattens(const matrix::CRS<double> &A,
const tensor::tensor_Dense<double> &x,
tensor::tensor_Dense<double> &y);
void mattens(const matrix::CRS<float> &A, const tensor::tensor_Dense<float> &x,
tensor::tensor_Dense<float> &y);




void mattens(const double &a, const matrix::CRS<double> &A,
const tensor::tensor_Dense<double> &x, const double &b,
tensor::tensor_Dense<double> &y);
void mattens(const float &a, const matrix::CRS<float> &A,
const tensor::tensor_Dense<float> &x, const float &b,
tensor::tensor_Dense<float> &y);



} 
} 
