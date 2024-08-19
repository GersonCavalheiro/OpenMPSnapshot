#pragma once

#include "cuda/memory.hpp"

namespace bnmf_algs {
namespace details {
namespace bld_mult {

template <typename Real>
void update_grad_plus_cuda(const cuda::DeviceMemory3D<Real>& S,
const cuda::DeviceMemory2D<Real>& beta_eph,
cuda::DeviceMemory3D<Real>& grad_plus);


template <typename Real>
void update_nom_cuda(const cuda::DeviceMemory2D<Real>& X_reciprocal,
const cuda::DeviceMemory2D<Real>& grad_minus,
const cuda::DeviceMemory3D<Real>& S,
cuda::DeviceMemory2D<Real>& nom);


template <typename Real>
void update_denom_cuda(const cuda::DeviceMemory2D<Real>& X_reciprocal,
const cuda::DeviceMemory3D<Real>& grad_plus,
const cuda::DeviceMemory3D<Real>& S,
cuda::DeviceMemory2D<Real>& denom);


template <typename Real>
void update_S_cuda(const cuda::DeviceMemory2D<Real>& X,
const cuda::DeviceMemory2D<Real>& nom,
const cuda::DeviceMemory2D<Real>& denom,
const cuda::DeviceMemory2D<Real>& grad_minus,
const cuda::DeviceMemory3D<Real>& grad_plus,
const cuda::DeviceMemory2D<Real>& S_ijp,
cuda::DeviceMemory3D<Real>& S);

} 
} 
} 
