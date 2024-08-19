#pragma once

#include <cmath>
#include <limits>
#include <utility>

#include "defs.hpp"


namespace bnmf_algs {

namespace nmf {

template <typename T, typename Real>
std::pair<matrix_t<T>, matrix_t<T>> nmf(const matrix_t<T>& X, size_t r,
Real beta, size_t max_iter = 1000) {
const auto x = static_cast<size_t>(X.rows());
const auto y = static_cast<size_t>(X.cols());
const auto z = r;

BNMF_ASSERT((X.array() >= 0).all(), "X matrix must be nonnegative");
BNMF_ASSERT(r > 0, "r must be positive");

if (X.isZero(0)) {
return std::make_pair(matrixd::Zero(x, z), matrixd::Zero(z, y));
}

matrix_t<T> W = matrix_t<T>::Random(x, z) + matrix_t<T>::Ones(x, z);
matrix_t<T> H = matrix_t<T>::Random(z, y) + matrix_t<T>::Ones(z, y);

if (max_iter == 0) {
return std::make_pair(W, H);
}

const Real p = 2 - beta;
const double eps = std::numeric_limits<double>::epsilon();
const auto p_int = static_cast<int>(p);
const bool p_is_int = std::abs(p_int - p) <= eps;

matrix_t<T> X_hat, nom_W(x, z), denom_W(x, z), nom_H(z, y), denom_H(z, y);
while (max_iter-- > 0) {
X_hat = W * H;

if (p_is_int && p == 0) {
nom_W = X * H.transpose();
denom_W = X_hat * H.transpose();
} else if (p_is_int && p == 1) {
nom_W = (X.array() / X_hat.array()).matrix() * H.transpose();
denom_W = matrixd::Ones(x, y) * H.transpose();
} else if (p_is_int && p == 2) {
nom_W =
(X.array() / X_hat.array().square()).matrix() * H.transpose();
denom_W = (X_hat.array().inverse()).matrix() * H.transpose();
} else {
nom_W = (X.array() / X_hat.array().pow(p)).matrix() * H.transpose();
denom_W = (X_hat.array().pow(1 - p)).matrix() * H.transpose();
}
W = W.array() * nom_W.array() / denom_W.array();

X_hat = W * H;

if (p_is_int && p == 0) {
nom_H = W.transpose() * X;
denom_H = W.transpose() * X_hat;
} else if (p_is_int && p == 1) {
nom_H = W.transpose() * (X.array() / X_hat.array()).matrix();
denom_H = W.transpose() * matrixd::Ones(x, y);
} else if (p_is_int && p == 2) {
nom_H =
W.transpose() * (X.array() / X_hat.array().square()).matrix();
denom_H = W.transpose() * (X_hat.array().inverse()).matrix();
} else {
nom_H = W.transpose() * (X.array() / X_hat.array().pow(p)).matrix();
denom_H = W.transpose() * (X_hat.array().pow(1 - p)).matrix();
}
H = H.array() * nom_H.array() / denom_H.array();
}
return std::make_pair(W, H);
}


template <typename Real>
Real beta_divergence(Real x, Real y, Real beta, double eps = 1e-50) {
if (std::abs(beta) <= std::numeric_limits<double>::epsilon()) {
if (std::abs(x) <= std::numeric_limits<double>::epsilon()) {
return -1;
}
return x / (y + eps) - std::log(x / (y + eps)) - 1;
} else if (std::abs(beta - 1) <= std::numeric_limits<double>::epsilon()) {
Real logpart;
if (std::abs(x) <= std::numeric_limits<double>::epsilon()) {
logpart = 0;
} else {
logpart = x * std::log(x / (y + eps));
}
return logpart - x + y;
}

Real nom = std::pow(x, beta) + (beta - 1) * std::pow(y, beta) -
beta * x * std::pow(y, beta - 1);
return nom / (beta * (beta - 1));
}


template <typename Tensor>
typename Tensor::value_type beta_divergence(const Tensor& X, const Tensor& Y,
typename Tensor::value_type beta,
double eps = 1e-50) {
typename Tensor::value_type result = 0;
#pragma omp parallel for schedule(static) reduction(+:result)
for (long i = 0; i < X.rows(); ++i) {
for (long j = 0; j < X.cols(); ++j) {
result += beta_divergence(X(i, j), Y(i, j), beta, eps);
}
}

return result;
}


template <typename InputIterator1, typename InputIterator2>
auto beta_divergence_seq(InputIterator1 first_begin, InputIterator1 first_end,
InputIterator2 second_begin,
decltype(*first_begin) beta, double eps = 1e-50) {
return std::inner_product(first_begin, first_end, second_begin, 0.0,
std::plus<>(), [beta, eps](double x, double y) {
return beta_divergence(x, y, beta, eps);
});
}
} 
} 
