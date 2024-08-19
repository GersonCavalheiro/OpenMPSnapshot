#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util/sampling.hpp"
#include "util_details.hpp"

namespace bnmf_algs {
namespace details {

template <typename T, typename Scalar> class CollapsedGibbsComputer {
public:

explicit CollapsedGibbsComputer(
const matrix_t<T>& X, size_t z,
const alloc_model::Params<Scalar>& model_params, size_t max_iter,
double eps)
: model_params(model_params),
one_sampler_repl(util::sample_ones_replace(X, max_iter)),
one_sampler_no_repl(util::sample_ones_noreplace(X)),
U_ipk(matrix_t<T>::Zero(X.rows(), z)), U_ppk(vector_t<T>::Zero(z)),
U_pjk(matrix_t<T>::Zero(X.cols(), z)),
sum_alpha(std::accumulate(model_params.alpha.begin(),
model_params.alpha.end(), Scalar())),
eps(eps), rnd_gen(util::gsl_rng_wrapper(gsl_rng_alloc(gsl_rng_taus),
gsl_rng_free)) {}


void operator()(size_t curr_step, tensor_t<T, 3>& S_prev) {
size_t i, j;
if (curr_step == 0) {
for (const auto& pair : one_sampler_no_repl) {
std::tie(i, j) = pair;
increment_sampling(i, j, S_prev);
}
} else {
std::tie(i, j) = *one_sampler_repl.begin();
++one_sampler_repl.begin();
decrement_sampling(i, j, S_prev);
increment_sampling(i, j, S_prev);
}
}

private:

void increment_sampling(size_t i, size_t j, tensor_t<T, 3>& S_prev) {
vector_t<T> prob(U_ppk.cols());
Eigen::Map<vector_t<T>> beta(model_params.beta.data(), 1,
model_params.beta.size());
std::vector<unsigned int> multinomial_sample(
static_cast<unsigned long>(U_ppk.cols()));

vector_t<T> alpha_row = U_ipk.row(i).array() + model_params.alpha[i];
vector_t<T> beta_row = (U_pjk.row(j) + beta).array();
vector_t<T> sum_alpha_row = U_ppk.array() + sum_alpha;
prob = alpha_row.array() * beta_row.array() /
(sum_alpha_row.array() + eps);

gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
prob.data(), multinomial_sample.data());
auto k = std::distance(multinomial_sample.begin(),
std::max_element(multinomial_sample.begin(),
multinomial_sample.end()));

++S_prev(i, j, k);
++U_ipk(i, k);
++U_ppk(k);
++U_pjk(j, k);
}


void decrement_sampling(size_t i, size_t j, tensor_t<T, 3>& S_prev) {
vector_t<T> prob(U_ppk.cols());
Eigen::Map<vector_t<T>> beta(model_params.beta.data(), 1,
model_params.beta.size());
std::vector<unsigned int> multinomial_sample(
static_cast<unsigned long>(U_ppk.cols()));

for (long k = 0; k < S_prev.dimension(2); ++k) {
prob(k) = S_prev(i, j, k);
}

gsl_ran_multinomial(rnd_gen.get(), static_cast<size_t>(prob.cols()), 1,
prob.data(), multinomial_sample.data());
auto k = std::distance(multinomial_sample.begin(),
std::max_element(multinomial_sample.begin(),
multinomial_sample.end()));

--S_prev(i, j, k);
--U_ipk(i, k);
--U_ppk(k);
--U_pjk(j, k);
}

private:
alloc_model::Params<Scalar> model_params;
private:
util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer<T>>
one_sampler_repl;
util::Generator<std::pair<int, int>,
details::SampleOnesNoReplaceComputer<T>>
one_sampler_no_repl;
matrix_t<T> U_ipk;
vector_t<T> U_ppk;
matrix_t<T> U_pjk;
T sum_alpha;
double eps;
util::gsl_rng_wrapper rnd_gen;
};
} 

namespace bld {

template <typename T, typename Scalar>
util::Generator<tensor_t<T, 3>, details::CollapsedGibbsComputer<T, Scalar>>
collapsed_gibbs(const matrix_t<T>& X, size_t z,
const alloc_model::Params<Scalar>& model_params,
size_t max_iter = 1000, double eps = 1e-50) {
details::check_bld_params(X, z, model_params);

tensor_t<T, 3> init_val(X.rows(), X.cols(), z);
init_val.setZero();

util::Generator<tensor_t<T, 3>, details::CollapsedGibbsComputer<T, Scalar>>
gen(init_val, max_iter + 2,
details::CollapsedGibbsComputer<T, Scalar>(X, z, model_params,
max_iter, eps));

++gen.begin();

return gen;
}
} 
} 
