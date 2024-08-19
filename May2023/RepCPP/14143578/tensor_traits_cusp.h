#pragma once

#include <cassert>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include "vector_categories.h"
#include "matrix_categories.h"
#include "tensor_traits.h"

namespace dg
{
template<class T>
struct TensorTraits<cusp::array1d<T,cusp::host_memory>,
std::enable_if_t< std::is_arithmetic<T>::value>>
{
using value_type        = T;
using tensor_category   = CuspVectorTag;
using execution_policy  = SerialTag;
};
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CPP
template<class T>
struct TensorTraits<cusp::array1d<T,cusp::device_memory>,
std::enable_if_t< std::is_arithmetic<T>::value>>
{
using value_type        = T;
using tensor_category   = CuspVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
using execution_policy  = CudaTag ;
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
using execution_policy  = OmpTag ;
#endif
};
#endif

template< class I, class V, class M>
struct TensorTraits< cusp::coo_matrix<I,V,M> >
{
using value_type = V;
using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::csr_matrix<I,V,M> >
{
using value_type = V;
using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::dia_matrix<I,V,M> >
{
using value_type = V;
using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::ell_matrix<I,V,M> >
{
using value_type = V;
using tensor_category = CuspMatrixTag;
};
template< class I, class V, class M>
struct TensorTraits< cusp::hyb_matrix<I,V,M> >
{
using value_type = V;
using tensor_category = CuspMatrixTag;
};


} 
