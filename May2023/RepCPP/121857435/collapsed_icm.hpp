#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/sampling.hpp"
#include "util_details.hpp"

namespace bnmf_algs {
namespace bld {

template <typename T, typename Scalar>
tensor_t<T, 3> collapsed_icm(const matrix_t<T>& X, size_t z,
const alloc_model::Params<Scalar>& model_params,
size_t max_iter = 1000, double eps = 1e-50) {
details::check_bld_params(X, z, model_params);

const auto x = static_cast<size_t>(X.rows());
const auto y = static_cast<size_t>(X.cols());

tensor_t<T, 3> U(x, y, z);
U.setZero();

matrix_t<T> U_ipk = matrix_t<T>::Zero(x, z);
vector_t<T> U_ppk = vector_t<T>::Zero(z);
matrix_t<T> U_pjk = matrix_t<T>::Zero(y, z);

const Scalar sum_alpha = std::accumulate(model_params.alpha.begin(),
model_params.alpha.end(), 0.0);

vector_t<T> prob(U_ppk.cols());
Eigen::Map<const vector_t<Scalar>> beta(model_params.beta.data(), 1,
model_params.beta.size());

size_t i, j;
for (const auto& pair : util::sample_ones_noreplace(X)) {
std::tie(i, j) = pair;

vector_t<T> alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
vector_t<T> beta_row = (U_pjk.row(j) + beta).array();
vector_t<T> sum_alpha_row = U_ppk.array() + sum_alpha;
prob = alpha_row.array() * beta_row.array() /
(sum_alpha_row.array() + eps);

auto k = std::distance(
prob.data(),
std::max_element(prob.data(), prob.data() + prob.cols()));

++U(i, j, k);
++U_ipk(i, k);
++U_ppk(k);
++U_pjk(j, k);
}

std::vector<unsigned int> multinomial_sample(
static_cast<unsigned long>(U_ppk.cols()));
util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);

for (const auto& pair : util::sample_ones_replace(X, max_iter)) {
std::tie(i, j) = pair;

for (long k = 0; k < U.dimension(2); ++k) {
prob(k) = U(i, j, k);
}

gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
prob.data(), multinomial_sample.data());
auto k = std::distance(multinomial_sample.begin(),
std::max_element(multinomial_sample.begin(),
multinomial_sample.end()));

--U(i, j, k);
--U_ipk(i, k);
--U_ppk(k);
--U_pjk(j, k);

vector_t<T> alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
vector_t<T> beta_row = (U_pjk.row(j) + beta).array();
vector_t<T> sum_alpha_row = U_ppk.array() + sum_alpha;
prob = alpha_row.array() * beta_row.array() /
(sum_alpha_row.array() + eps);

k = std::distance(
prob.data(),
std::max_element(prob.data(), prob.data() + prob.cols()));

++U(i, j, k);
++U_ipk(i, k);
++U_ppk(k);
++U_pjk(j, k);
}

return U;
}
} 
} 
