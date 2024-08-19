#pragma once

#include "defs.hpp"
#include "util/sampling.hpp"
#include "util_details.hpp"

namespace bnmf_algs {
namespace bld {

template <typename T, typename Scalar>
tensor_t<T, 3> seq_greedy_bld(const matrix_t<T>& X, size_t z,
const alloc_model::Params<Scalar>& model_params) {
details::check_bld_params(X, z, model_params);

long x = X.rows();
long y = X.cols();

if (std::abs(X.sum()) <= std::numeric_limits<double>::epsilon()) {
tensor_t<T, 3> S(x, y, z);
S.setZero();
return S;
}

tensor_t<T, 3> S(x, y, z);
S.setZero();
matrix_t<T> S_ipk = matrix_t<T>::Zero(x, z); 
vector_t<T> S_ppk = vector_t<T>::Zero(z);    
matrix_t<T> S_pjk = matrix_t<T>::Zero(y, z); 
const Scalar sum_alpha = std::accumulate(model_params.alpha.begin(),
model_params.alpha.end(), 0.0);

auto matrix_sampler = util::sample_ones_noreplace(X);
int ii, jj;
for (const auto& sample : matrix_sampler) {
std::tie(ii, jj) = sample;

size_t kmax = 0;
T curr_max = std::numeric_limits<double>::lowest();
for (size_t k = 0; k < z; ++k) {
T log_marginal_change =
std::log(model_params.alpha[ii] + S_ipk(ii, k)) -
std::log(sum_alpha + S_ppk(k)) - std::log(1 + S(ii, jj, k)) +
std::log(model_params.beta[k] + S_pjk(jj, k));

if (log_marginal_change > curr_max) {
curr_max = log_marginal_change;
kmax = k;
}
}

++S(ii, jj, kmax);
++S_ipk(ii, kmax);
++S_ppk(kmax);
++S_pjk(jj, kmax);
}

return S;
}
} 
} 
