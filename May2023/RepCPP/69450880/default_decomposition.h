




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/internal/decompose.h>

namespace hydra_thrust
{
namespace system
{
namespace omp
{
namespace detail
{

template <typename IndexType>
hydra_thrust::system::detail::internal::uniform_decomposition<IndexType> default_decomposition(IndexType n);

} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/omp/detail/default_decomposition.inl>

