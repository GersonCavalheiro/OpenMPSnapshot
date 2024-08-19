#pragma once

#include "alloc_model/alloc_model_params.hpp"
#include "bld/util_details.hpp"
#include "bld_mult_funcs.hpp"
#include "defs.hpp"
#include "util/util.hpp"
#include <gsl/gsl_sf_psi.h>

#ifdef USE_OPENMP
#include <thread>
#endif

#ifdef USE_CUDA
#include "bld_mult_cuda_funcs.hpp"
#include "cuda/memory.hpp"
#include "cuda/tensor_ops.hpp"
#endif

namespace bnmf_algs {
namespace bld {

template <typename T, typename Scalar>
tensor_t<T, 3> bld_mult(const matrix_t<T>& X, const size_t z,
const alloc_model::Params<Scalar>& model_params,
size_t max_iter = 1000, bool use_psi_appr = false,
double eps = 1e-50) {
details::check_bld_params(X, z, model_params);

const auto x = static_cast<size_t>(X.rows());
const auto y = static_cast<size_t>(X.cols());

tensor_t<T, 3> S = details::bld_mult::init_S(X, z);
const matrix_t<T> X_reciprocal = details::bld_mult::X_reciprocal(X, eps);

tensor_t<T, 3> grad_plus(x, y, z);
matrix_t<T> nom_mult(x, y);
matrix_t<T> denom_mult(x, y);
matrix_t<T> grad_minus(x, z);
matrix_t<T> alpha_eph(x, z);
vector_t<T> alpha_eph_sum(z);
matrix_t<T> beta_eph(y, z);
tensor_t<T, 2> S_pjk(y, z);
tensor_t<T, 2> S_ipk(x, z);
tensor_t<T, 2> S_ijp(x, y);
vector_t<T> alpha;
vector_t<T> beta;

std::tie(alpha, beta) = details::bld_mult::init_alpha_beta(model_params);

#ifdef USE_OPENMP
Eigen::ThreadPool tp(std::thread::hardware_concurrency());
Eigen::ThreadPoolDevice thread_dev(&tp,
std::thread::hardware_concurrency());
#endif

const auto psi_fn =
use_psi_appr ? util::psi_appr<T> : details::gsl_psi_wrapper<T>;

for (size_t eph = 0; eph < max_iter; ++eph) {
#ifdef USE_OPENMP
S_pjk.device(thread_dev) = S.sum(shape<1>({0}));
S_ipk.device(thread_dev) = S.sum(shape<1>({1}));
S_ijp.device(thread_dev) = S.sum(shape<1>({2}));
#else
S_pjk = S.sum(shape<1>({0}));
S_ipk = S.sum(shape<1>({1}));
S_ijp = S.sum(shape<1>({2}));
#endif

details::bld_mult::update_alpha_eph(S_ipk, alpha, alpha_eph);
details::bld_mult::update_beta_eph(S_pjk, beta, beta_eph);

details::bld_mult::update_grad_plus(S, beta_eph, psi_fn, grad_plus);
details::bld_mult::update_grad_minus(alpha_eph, psi_fn, grad_minus);

details::bld_mult::update_nom_mult(X_reciprocal, grad_minus, S,
nom_mult);
details::bld_mult::update_denom_mult(X_reciprocal, grad_plus, S,
denom_mult);
details::bld_mult::update_S(X, nom_mult, denom_mult, grad_minus,
grad_plus, S_ijp, S, eps);
}

return S;
}

#ifdef USE_CUDA


template <typename T, typename Scalar>
tensor_t<T, 3> bld_mult_cuda(const matrix_t<T>& X, const size_t z,
const alloc_model::Params<Scalar>& model_params,
size_t max_iter = 1000, bool use_psi_appr = false,
double eps = 1e-50) {
details::check_bld_params(X, z, model_params);

const auto x = static_cast<size_t>(X.rows());
const auto y = static_cast<size_t>(X.cols());

tensor_t<T, 3> S = details::bld_mult::init_S(X, z);

const matrix_t<T> X_reciprocal = details::bld_mult::X_reciprocal(X, eps);

matrix_t<T> grad_minus(x, z);
matrix_t<T> alpha_eph(x, z);
vector_t<T> alpha_eph_sum(z);
matrix_t<T> beta_eph(y, z);
tensor_t<T, 2> S_pjk(y, z);
tensor_t<T, 2> S_ipk(x, z);
tensor_t<T, 2> S_ijp(x, y);
vector_t<T> alpha;
vector_t<T> beta;

std::tie(alpha, beta) = details::bld_mult::init_alpha_beta(model_params);

cuda::HostMemory3D<T> S_host(S.data(), x, y, z);
cuda::HostMemory2D<T> S_pjk_host(S_pjk.data(), y, z);
cuda::HostMemory2D<T> S_ipk_host(S_ipk.data(), x, z);
cuda::HostMemory2D<T> beta_eph_host(beta_eph.data(), y, z);
cuda::HostMemory2D<T> grad_minus_host(grad_minus.data(), x, z);

cuda::DeviceMemory2D<T> X_device(x, y);
cuda::DeviceMemory2D<T> X_reciprocal_device(x, y);
cuda::DeviceMemory3D<T> S_device(x, y, z);
std::array<cuda::DeviceMemory2D<T>, 3> device_sums = {
cuda::DeviceMemory2D<T>(y, z), cuda::DeviceMemory2D<T>(x, z),
cuda::DeviceMemory2D<T>(x, y)};
cuda::DeviceMemory3D<T> grad_plus_device(x, y, z);
cuda::DeviceMemory2D<T> beta_eph_device(y, z);
cuda::DeviceMemory2D<T> nom_device(x, y);
cuda::DeviceMemory2D<T> denom_device(x, y);
cuda::DeviceMemory2D<T> grad_minus_device(x, z);

cuda::copy3D(S_device, S_host);
{
cuda::HostMemory2D<const T> X_host(X.data(), x, y);
cuda::HostMemory2D<const T> X_reciprocal_host(X_reciprocal.data(), x,
y);
cuda::copy2D(X_device, X_host);
cuda::copy2D(X_reciprocal_device, X_reciprocal_host);
}

const auto psi_fn = use_psi_appr ? util::psi_appr<T> : gsl_sf_psi;

for (size_t eph = 0; eph < max_iter; ++eph) {
cuda::tensor_sums(S_device, device_sums);
cuda::copy2D(S_pjk_host, device_sums[0]);
cuda::copy2D(S_ipk_host, device_sums[1]);

details::bld_mult::update_beta_eph(S_pjk, beta, beta_eph);

cuda::copy2D(beta_eph_device, beta_eph_host);
details::bld_mult::update_grad_plus_cuda(S_device, beta_eph_device,
grad_plus_device);
details::bld_mult::update_denom_cuda(
X_reciprocal_device, grad_plus_device, S_device, denom_device);


details::bld_mult::update_alpha_eph(S_ipk, alpha, alpha_eph);
details::bld_mult::update_grad_minus(alpha_eph, psi_fn, grad_minus);

cuda::copy2D(grad_minus_device, grad_minus_host);
details::bld_mult::update_nom_cuda(
X_reciprocal_device, grad_minus_device, S_device, nom_device);

const auto& S_ijp_device = device_sums[2];
details::bld_mult::update_S_cuda(X_device, nom_device, denom_device,
grad_minus_device, grad_plus_device,
S_ijp_device, S_device);
}
cuda::copy3D(S_host, S_device);
return S;
}

#endif
} 
} 
