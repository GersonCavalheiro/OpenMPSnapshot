#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "defs.hpp"
#include "util_details.hpp"
#include "util/util.hpp"
#include <gsl/gsl_sf_psi.h>

namespace bnmf_algs {
namespace bld {

template <typename T, typename Scalar>
tensor_t<T, 3> bld_add(const matrix_t<T>& X, size_t z,
const alloc_model::Params<Scalar>& model_params,
size_t max_iter = 1000, double eps = 1e-50) {
details::check_bld_params(X, z, model_params);

long x = X.rows(), y = X.cols();
tensor_t<T, 3> S(x, y, z);
{
util::gsl_rng_wrapper rand_gen(gsl_rng_alloc(gsl_rng_taus),
gsl_rng_free);
std::vector<double> dirichlet_params(z, 1);
std::vector<double> dirichlet_variates(z);
for (int i = 0; i < x; ++i) {
for (int j = 0; j < y; ++j) {
gsl_ran_dirichlet(rand_gen.get(), z, dirichlet_params.data(),
dirichlet_variates.data());
for (size_t k = 0; k < z; ++k) {
S(i, j, k) = X(i, j) * dirichlet_variates[k];
}
}
}
}

tensor_t<T, 2> S_pjk(y, z); 
tensor_t<T, 2> S_ipk(x, z); 
matrix_t<T> alpha_eph(x, z);
matrix_t<T> beta_eph(y, z);
tensor_t<T, 3> grad_S(x, y, z);
tensor_t<T, 3> eps_tensor(x, y, z);
tensor_t<T, 3> Z(x, y, z);
tensor_t<T, 2> S_mult(x, y);
eps_tensor.setConstant(eps);

const auto eta = [](const size_t step) -> double {
return 0.1 / std::pow(step + 1, 0.55);
};

#ifdef USE_OPENMP
Eigen::ThreadPool tp(std::thread::hardware_concurrency());
Eigen::ThreadPoolDevice thread_dev(&tp,
std::thread::hardware_concurrency());
#endif

for (size_t eph = 0; eph < max_iter; ++eph) {
#ifdef USE_OPENMP
S_pjk.device(thread_dev) = S.sum(shape<1>({0}));
S_ipk.device(thread_dev) = S.sum(shape<1>({1}));
#else
S_pjk = S.sum(shape<1>({0}));
S_ipk = S.sum(shape<1>({1}));
#endif
#pragma omp parallel for schedule(static)
for (size_t k = 0; k < z; ++k) {
for (int i = 0; i < x; ++i) {
alpha_eph(i, k) = model_params.alpha[i] + S_ipk(i, k);
}
for (int j = 0; j < y; ++j) {
beta_eph(j, k) = model_params.beta[k] + S_pjk(j, k);
}
}

vector_t<T> alpha_eph_sum = alpha_eph.colwise().sum();
#pragma omp parallel for schedule(static)
for (int i = 0; i < x; ++i) {
for (int j = 0; j < y; ++j) {
for (size_t k = 0; k < z; ++k) {
grad_S(i, j, k) = util::psi_appr(beta_eph(j, k)) -
util::psi_appr(S(i, j, k) + 1) -
util::psi_appr(alpha_eph_sum(k)) +
util::psi_appr(alpha_eph(i, k));
}
}
}
#ifdef USE_OPENMP
Z.device(thread_dev) = S.log() + eps_tensor;
S_mult.device(thread_dev) = (S * grad_S).sum(shape<1>({2}));
#else
Z = S.log() + eps_tensor;
S_mult = (S * grad_S).sum(shape<1>({2}));
#endif

#pragma omp parallel for schedule(static)
for (int i = 0; i < x; ++i) {
for (int j = 0; j < y; ++j) {
for (size_t k = 0; k < z; ++k) {
Z(i, j, k) += eta(eph) * S(i, j, k) *
(X(i, j) * grad_S(i, j, k) - S_mult(i, j));
}
}
}
#ifdef USE_OPENMP
Z.device(thread_dev) = Z.exp();
#else
Z = Z.exp();
#endif
bnmf_algs::util::normalize(Z, 2, bnmf_algs::util::NormType::L1);

#pragma omp parallel for schedule(static)
for (int i = 0; i < x; ++i) {
for (int j = 0; j < y; ++j) {
for (size_t k = 0; k < z; ++k) {
S(i, j, k) = X(i, j) * Z(i, j, k);
}
}
}
}
return S;
}
} 
} 
