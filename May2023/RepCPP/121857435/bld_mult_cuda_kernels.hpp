#pragma once
#include "bld_mult_cuda_funcs.hpp"

namespace bnmf_algs {
namespace details {
namespace bld_mult {

namespace kernel {

template <typename Real> __device__ Real psi_appr(Real x);


template <typename Real>
__global__ void update_grad_plus(cudaPitchedPtr S, const Real* beta_eph,
size_t pitch, cudaPitchedPtr grad_plus,
size_t width, size_t height, size_t depth);


template <typename Real>
__global__ void update_nom(cudaPitchedPtr S, const Real* X_reciprocal,
size_t X_reciprocal_pitch, const Real* grad_minus,
size_t grad_minus_pitch, Real* nom_mult,
size_t nom_mult_pitch, size_t width, size_t height,
size_t depth);


template <typename Real>
__global__ void update_denom(cudaPitchedPtr S, const Real* X_reciprocal,
size_t X_reciprocal_pitch,
cudaPitchedPtr grad_plus, Real* denom_mult,
size_t denom_mult_pitch, size_t width,
size_t height, size_t depth);


template <typename Real>
__global__ void
update_S(const Real* X, size_t X_pitch, const Real* nom_mult,
size_t nom_mult_pitch, const Real* denom_mult, size_t denom_mult_pitch,
const Real* grad_minus, size_t grad_minus_pitch,
cudaPitchedPtr grad_plus, const Real* S_ijp, size_t S_ijp_pitch,
cudaPitchedPtr S, size_t width, size_t height, size_t depth);
} 
} 
} 
} 
