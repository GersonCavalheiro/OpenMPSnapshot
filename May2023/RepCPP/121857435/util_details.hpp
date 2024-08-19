#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include <gsl/gsl_sf_psi.h>
#include <vector>

namespace bnmf_algs {
namespace details {

template <typename T, typename Scalar>
void check_bld_params(const matrix_t<T>& X, size_t z,
const alloc_model::Params<Scalar>& model_params) {
BNMF_ASSERT((X.array() >= 0).all(), "X must be nonnegative");
BNMF_ASSERT(
model_params.alpha.size() == static_cast<size_t>(X.rows()),
"Number of alpha parameters must be equal to number of rows of X");
BNMF_ASSERT(model_params.beta.size() == static_cast<size_t>(z),
"Number of beta parameters must be equal to z");
}


template <typename T, typename Scalar>
void check_EM_params(
const matrix_t<T>& X,
const std::vector<alloc_model::Params<Scalar>>& param_vec) {

for (long i = 0; i < X.rows(); ++i) {
for (long j = 0; j < X.cols(); ++j) {
BNMF_ASSERT(std::isnan(X(i, j)) || X(i, j) >= 0,
"X must contain nonnegative values or NaN");
}
}

for (const auto& param : param_vec) {
BNMF_ASSERT(
param.alpha.size() == static_cast<size_t>(X.rows()),
"Number of alpha parameters must be equal to number of rows of X");
BNMF_ASSERT(param.beta.size() == static_cast<size_t>(X.cols()),
"Number of beta parameters must be equal to number of "
"columns of X");
}
}


template <typename Real> Real gsl_psi_wrapper(Real x) {
return static_cast<Real>(gsl_sf_psi(static_cast<double>(x)));
}
} 
} 
