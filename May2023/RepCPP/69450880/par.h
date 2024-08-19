

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator_aware_execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{


struct par_t : hydra_thrust::system::tbb::detail::execution_policy<par_t>,
hydra_thrust::detail::allocator_aware_execution_policy<
hydra_thrust::system::tbb::detail::execution_policy>
{
__host__ __device__
par_t() : hydra_thrust::system::tbb::detail::execution_policy<par_t>() {}
};


} 


static const detail::par_t par;


} 
} 


namespace tbb
{


using hydra_thrust::system::tbb::par;


} 
} 

