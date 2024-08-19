#pragma once

#include "Matrix.h"

namespace func2D{

template<typename T>
void abs(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin(); 
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = (*it_src > 0) ? (*it_src) : -(*it_src);
}
}

template<typename T>
void abs(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = (*it_src > 0) ? (*it_src) : -(*it_src);
}
}

template<typename T>
void pow(Matrix<T>& dest, const Matrix<T>& src, int power=2){
auto it_dest = dest.begin(); 
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src, ++it_dest)
*(it_dest) = std::pow((*it_src), power);
}

template<typename T>
void pow(Matrix<T>& src, int power=2){
Matrix<T> cpy = src;
auto it_src = src.begin(), it_cpy = cpy.begin();
for(; it_src != src.end(); ++it_src, ++it_cpy)
(*it_src) *= std::pow((*it_cpy), power);
}

template<typename T>
void sqrt(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = std::tanh((*it_src));
}
}

template<typename T>
void sqrt(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = std::sqrt((*it_src));
}
}

template<typename T>
void log(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin(); 
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = std::log(*it_src);
}
}

template<typename T>
void log(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = std::log(*it_src);
}
}

template<typename T>
void exp(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = std::exp(*it_src);
}
}

template<typename T>
void exp(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = std::exp(*it_src);
}
}

template<typename T>
void relu(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = std::max(static_cast<T>(0), *it_src); 
}
}

template<typename T>
void relu(Matrix<T>& src){
for(auto it_src = src.begin(); it_src != src.end(); ++it_src){
(*it_src) = std::max(static_cast<T>(0), *it_src); 
}
}

template<typename T>
void tanh(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = std::tanh((*it_src));
}
}

template<typename T>
void tanh(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = std::tanh((*it_src));
}
}

template<typename T>
void sigmoid(Matrix<T>& dest, const Matrix<T>& src){
T one = static_cast<T>(1);
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = one / (one + std::exp(-(*it_src)));
}
}

template<typename T>
void sigmoid(Matrix<T>& src){
T one = static_cast<T>(1);
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = one / (one + std::exp(-(*it_src)));
}
}

template<typename T>
void swish(Matrix<T>& dest, const Matrix<T>& src){
sigmoid(dest, src);
dest *= src;
}

template<typename T>
void swish(Matrix<T>& src){
T one = static_cast<T>(1);
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) *= one / (one + std::exp(-(*it_src)));
}
}
}

namespace deriv2D{

template<typename T>
void abs(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin(); 
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = (*it_src > 0) ? 1 : -1;
}
}
template<typename T>
void abs(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = (*it_src > 0) ? 1 : -1;
}
}
template<typename T>
void pow(Matrix<T>& dest, const Matrix<T>& src, int power=2){
auto it_src = src.begin();
auto it_dest = dest.begin();
int exponent = power-1;
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = power * std::pow(*it_src, exponent);
}
}
template<typename T>
void pow(Matrix<T>& src, int power=2){
auto it_src = src.begin();
int exponent = power-1;
for(; it_src != src.end(); ++it_src){
(*it_src) = power * std::pow(*it_src, exponent);
}
}
template<typename T>
void sqrt(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = 1 / (2 * std::sqrt(*it_src) + 1e-8f);
}
}
template<typename T>
void sqrt(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = 1 / (2 * std::sqrt(*it_src) + 1e-8f);
}
}
template<typename T>
void log(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = 1 / ((*it_src) + 1e-8f);
}
}
template<typename T>
void log(Matrix<T>& src){
T type_min = std::numeric_limits<T>::min();
T one = static_cast<T>(1);
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = one / ((*it_src) + type_min);
}
}
template<typename T>
void exp(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
int sgn = (static_cast<T>(0) < (*it_src)) - ((*it_src) < static_cast<T>(0));
(*it_dest) = sgn * std::exp(*it_src);
}
}
template<typename T>
void exp(Matrix<T>& src){
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
int sgn = (static_cast<T>(0) < (*it_src)) - ((*it_src) < static_cast<T>(0));
(*it_src) = sgn * std::exp(*it_src);
}
}
template<typename T>
void relu(Matrix<T>& dest, const Matrix<T>& src){
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = ((*it_src) > 0) ? 1 : 0;
}
}
template<typename T>
void relu(Matrix<T>& src){
for(auto it_src = src.begin(); it_src != src.end(); ++it_src){
(*it_src) = ((*it_src) > 0) ? 1 : 0;
}
}
template<typename T>
void tanh(Matrix<T>& dest, const Matrix<T>& src){
T one = static_cast<T>(1);
auto it_src = src.begin(), it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = one - (*it_src) * (*it_src);
}
}
template<typename T>
void tanh(Matrix<T>& src){
T one = static_cast<T>(1);
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = one - (*it_src) * (*it_src);
}
}
template<typename T>
void sigmoid(Matrix<T>& dest, const Matrix<T>& src){
T one = static_cast<T>(1);
auto it_src = src.begin();
auto it_dest = dest.begin();
for(; it_src != src.end(); ++it_src, ++it_dest){
(*it_dest) = (*it_src) * (one - (*it_src));
}
}
template<typename T>
void sigmoid(Matrix<T>& src){
T one = static_cast<T>(1);
auto it_src = src.begin();
for(; it_src != src.end(); ++it_src){
(*it_src) = (*it_src) * (one - (*it_src));
}
}
template<typename T>
void swish(Matrix<T>& dest, const Matrix<T>& src_x, const Matrix<T>& src_y){
T one = static_cast<T>(1);
Matrix<T> sig_mat(src_x.getRows(), src_x.getCols(), 0);
func2D::sigmoid(sig_mat, src_x);
sig_mat *= one + src_y * (-one);
dest = src_y + sig_mat;
}
}