#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/generator.hpp"
#include "util/util.hpp"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <tuple>
#include <vector>

namespace bnmf_algs {
namespace alloc_model {
template <typename T, typename Scalar>
double log_marginal_S(const tensor_t<T, 3>& S,
const Params<Scalar>& model_params);
} 

namespace details {


template <typename T, typename Scalar>
double compute_first_term(const tensor_t<T, 3>& S,
const std::vector<Scalar>& alpha) {
const long x = S.dimension(0);
const long y = S.dimension(1);
const long z = S.dimension(2);

const double log_gamma_sum =
gsl_sf_lngamma(std::accumulate(alpha.begin(), alpha.end(), 0.0));

std::vector<double> log_gamma_alpha(alpha.size());
std::transform(
alpha.begin(), alpha.end(), log_gamma_alpha.begin(),
[](const Scalar alpha_i) { return gsl_sf_lngamma(alpha_i); });

const double sum_log_gamma =
std::accumulate(log_gamma_alpha.begin(), log_gamma_alpha.end(), 0.0);

double first = 0;
#pragma omp parallel for reduction(+:first)
for (int k = 0; k < z; ++k) {
double sum = 0;
for (int i = 0; i < x; ++i) {
sum += alpha[i];
for (int j = 0; j < y; ++j) {
sum += S(i, j, k);
}
}
first -= gsl_sf_lngamma(sum);

for (int i = 0; i < x; ++i) {
sum = alpha[i];
for (int j = 0; j < y; ++j) {
sum += S(i, j, k);
}
first += gsl_sf_lngamma(sum);
}
}

double base = log_gamma_sum * z - sum_log_gamma * z;

return base + first;
}


template <typename T, typename Scalar>
double compute_second_term(const tensor_t<T, 3>& S,
const std::vector<Scalar>& beta) {
const long x = S.dimension(0);
const long y = S.dimension(1);
const long z = S.dimension(2);

const double log_gamma_sum =
gsl_sf_lngamma(std::accumulate(beta.begin(), beta.end(), 0.0));

std::vector<double> log_gamma_beta(beta.size());
std::transform(beta.begin(), beta.end(), log_gamma_beta.begin(),
[](const Scalar beta) { return gsl_sf_lngamma(beta); });

const double sum_log_gamma =
std::accumulate(log_gamma_beta.begin(), log_gamma_beta.end(), 0.0);

double second = 0;
#pragma omp parallel for reduction(+:second)
for (int j = 0; j < y; ++j) {
double sum = 0;
for (int k = 0; k < z; ++k) {
sum += beta[k];
for (int i = 0; i < x; ++i) {
sum += S(i, j, k);
}
}
second -= gsl_sf_lngamma(sum);

for (int k = 0; k < z; ++k) {
sum = beta[k];
for (int i = 0; i < x; ++i) {
sum += S(i, j, k);
}
second += gsl_sf_lngamma(sum);
}
}

double base = log_gamma_sum * y - sum_log_gamma * y;
return base + second;
}


template <typename T, typename Scalar>
double compute_third_term(const tensor_t<T, 3>& S, Scalar a, Scalar b) {
const long x = S.dimension(0);
const long y = S.dimension(1);
const long z = S.dimension(2);

const double log_gamma = -gsl_sf_lngamma(a);
const double a_log_b = a * std::log(b);

double third = 0;
#pragma omp parallel for reduction(+:third)
for (int j = 0; j < y; ++j) {
double sum = a;
for (int i = 0; i < x; ++i) {
for (int k = 0; k < z; ++k) {
sum += S(i, j, k);
}
}
third += gsl_sf_lngamma(sum);
third -= sum * std::log(b + 1);
}

double base = log_gamma * y + a_log_b * y;
return base + third;
}


template <typename T> double compute_fourth_term(const tensor_t<T, 3>& S) {
const long x = S.dimension(0);
const long y = S.dimension(1);
const long z = S.dimension(2);

double fourth = 0;
#pragma omp parallel for reduction(+:fourth)
for (int i = 0; i < x; ++i) {
for (int j = 0; j < y; ++j) {
for (int k = 0; k < z; ++k) {
fourth += gsl_sf_lngamma(S(i, j, k) + 1);
}
}
}
return fourth;
}


template <typename Integer, typename Scalar> class TotalMarginalCalculator {
public:

TotalMarginalCalculator(const matrix_t<Integer>& X,
const alloc_model::Params<Scalar>& model_params)
: X(X), model_params(model_params),
S(X.rows(), X.cols(), model_params.beta.size()) {

const Integer z = model_params.beta.size();
for (long i = 0; i < X.rows(); ++i) {
for (long j = 0; j < X.cols(); ++j) {
if (X(i, j) != 0) {
ii.push_back(i);
jj.push_back(j);
alloc_vec.push_back(
util::partition_change_indices(X(i, j), z));
}
}
}

S.setZero();
for (long i = 0; i < X.rows(); ++i) {
for (long j = 0; j < X.cols(); ++j) {
S(i, j, 0) = X(i, j);
}
}
S_pjk = S.sum(shape<1>({0}));
S_ipk = S.sum(shape<1>({1}));
S_ppk = S.sum(shape<2>({0, 1}));

sum_alpha = std::accumulate(model_params.alpha.begin(),
model_params.alpha.end(), Scalar());
}


double calc_marginal() {
const double init_log_marginal =
alloc_model::log_marginal_S(S, model_params);

return marginal_recursive(0, init_log_marginal);
}

private:

double log_marginal_change_on_increment(size_t i, size_t j, size_t k) {
return std::log(model_params.alpha[i] + S_ipk(i, k)) -
std::log(sum_alpha + S_ppk(k)) - std::log(1 + S(i, j, k)) +
std::log(model_params.beta[k] + S_pjk(j, k));
}


double log_marginal_change_on_decrement(size_t i, size_t j, size_t k) {
double result = -(std::log(model_params.alpha[i] + S_ipk(i, k) - 1) -
std::log(sum_alpha + S_ppk(k) - 1) -
std::log(1 + S(i, j, k) - 1) +
std::log(model_params.beta[k] + S_pjk(j, k) - 1));

return result;
}


double marginal_recursive(const size_t fiber_index,
double prev_log_marginal) {

const auto& part_changes = alloc_vec[fiber_index];
const size_t i = ii[fiber_index];
const size_t j = jj[fiber_index];

const Integer old_value = S(i, j, 0);

const bool last_fiber = fiber_index == (ii.size() - 1);

double result =
last_fiber ? std::exp(prev_log_marginal)
: marginal_recursive(fiber_index + 1, prev_log_marginal);

size_t incr_idx, decr_idx;
double increment_change, decrement_change, new_log_marginal;
for (const auto& change_idx : part_changes) {
std::tie(decr_idx, incr_idx) = change_idx;

decrement_change = log_marginal_change_on_decrement(i, j, decr_idx);

--S(i, j, decr_idx);
--S_pjk(j, decr_idx);
--S_ipk(i, decr_idx);
--S_ppk(decr_idx);

increment_change = log_marginal_change_on_increment(i, j, incr_idx);

++S(i, j, incr_idx);
++S_pjk(j, incr_idx);
++S_ipk(i, incr_idx);
++S_ppk(incr_idx);

new_log_marginal =
prev_log_marginal + increment_change + decrement_change;

if (last_fiber) {
result += std::exp(new_log_marginal);
} else {
result += marginal_recursive(fiber_index + 1, new_log_marginal);
}

prev_log_marginal = new_log_marginal;
}

const auto last_index = S.dimension(2) - 1;
S(i, j, last_index) = 0;
S(i, j, 0) = old_value;
S_pjk(j, last_index) -= old_value;
S_pjk(j, 0) += old_value;
S_ipk(i, last_index) -= old_value;
S_ipk(i, 0) += old_value;
S_ppk(last_index) -= old_value;
S_ppk(0) += old_value;

return result;
}

private:

const matrix_t<Integer>& X;

const alloc_model::Params<Scalar>& model_params;

tensor_t<Integer, 3> S;

tensor_t<Integer, 2> S_pjk;

tensor_t<Integer, 2> S_ipk;

tensor_t<Integer, 1> S_ppk;

Scalar sum_alpha;

std::vector<size_t> ii;

std::vector<size_t> jj;

std::vector<std::vector<std::pair<size_t, size_t>>> alloc_vec;
};
} 


namespace alloc_model {


template <typename T, typename Scalar>
std::tuple<matrix_t<T>, matrix_t<T>, vector_t<T>>
bnmf_priors(const shape<3>& tensor_shape, const Params<Scalar>& model_params) {
size_t x = tensor_shape[0], y = tensor_shape[1], z = tensor_shape[2];

BNMF_ASSERT(model_params.alpha.size() == x,
"Number of dirichlet parameters alpha must be equal to x");
BNMF_ASSERT(model_params.beta.size() == z,
"Number of dirichlet parameters beta must be equal to z");

util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);

vector_t<T> prior_L(y);
for (size_t i = 0; i < y; ++i) {
prior_L(i) =
gsl_ran_gamma(rand_gen.get(), model_params.a, model_params.b);
}

matrix_t<T> prior_W(x, z);
vector_t<T> dirichlet_variates(x);
for (size_t i = 0; i < z; ++i) {
gsl_ran_dirichlet(rand_gen.get(), x, model_params.alpha.data(),
dirichlet_variates.data());

for (size_t j = 0; j < x; ++j) {
prior_W(j, i) = dirichlet_variates(j);
}
}

matrix_t<T> prior_H(z, y);
dirichlet_variates = vector_t<T>(z);
for (size_t i = 0; i < y; ++i) {
gsl_ran_dirichlet(rand_gen.get(), z, model_params.beta.data(),
dirichlet_variates.data());

for (size_t j = 0; j < z; ++j) {
prior_H(j, i) = dirichlet_variates(j);
}
}

return std::make_tuple(prior_W, prior_H, prior_L);
}


template <typename T>
tensor_t<T, 3> sample_S(const matrix_t<T>& prior_W, const matrix_t<T>& prior_H,
const vector_t<T>& prior_L) {
auto x = static_cast<size_t>(prior_W.rows());
auto y = static_cast<size_t>(prior_L.cols());
auto z = static_cast<size_t>(prior_H.rows());

BNMF_ASSERT(prior_W.cols() == prior_H.rows(),
"Number of columns of W is different than number of rows of H");
BNMF_ASSERT(prior_H.cols() == prior_L.cols(),
"Number of columns of H is different than size of L");

util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);
tensor_t<T, 3> sample(x, y, z);
double mu;
for (size_t i = 0; i < x; ++i) {
for (size_t j = 0; j < y; ++j) {
for (size_t k = 0; k < z; ++k) {
mu = prior_W(i, k) * prior_H(k, j) * prior_L(j);
sample(i, j, k) = gsl_ran_poisson(rand_gen.get(), mu);
}
}
}
return sample;
}


template <typename T, typename Scalar>
double log_marginal_S(const tensor_t<T, 3>& S,
const Params<Scalar>& model_params) {
BNMF_ASSERT(model_params.alpha.size() ==
static_cast<size_t>(S.dimension(0)),
"Number of alpha parameters must be equal to S.dimension(0)");
BNMF_ASSERT(model_params.beta.size() == static_cast<size_t>(S.dimension(2)),
"Number of alpha parameters must be equal to S.dimension(2)");

return details::compute_first_term(S, model_params.alpha) +
details::compute_second_term(S, model_params.beta) +
details::compute_third_term(S, model_params.a, model_params.b) -
details::compute_fourth_term(S);
}


template <typename Integer, typename Scalar>
double total_log_marginal(const matrix_t<Integer>& X,
const Params<Scalar>& model_params) {
BNMF_ASSERT((X.array() >= 0).all(),
"X must be nonnegative in alloc_model::total_log_marginal");
BNMF_ASSERT(static_cast<size_t>(X.rows()) == model_params.alpha.size(),
"Model parameters are incompatible with given matrix X in "
"alloc_model::total_log_marginal");

details::TotalMarginalCalculator<Integer, Scalar> calc(X, model_params);
double marginal = calc.calc_marginal();

return std::log(marginal);
}

} 
} 
