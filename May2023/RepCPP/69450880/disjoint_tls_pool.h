



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/mr/disjoint_pool.h>

namespace hydra_thrust
{
namespace mr
{




template<typename Upstream, typename Bookkeeper>
__host__ __device__
hydra_thrust::mr::disjoint_unsynchronized_pool_resource<Upstream, Bookkeeper> & tls_disjoint_pool(
Upstream * upstream = NULL,
Bookkeeper * bookkeeper = NULL)
{
static thread_local auto adaptor = [&]{
assert(upstream && bookkeeper);
return hydra_thrust::mr::disjoint_unsynchronized_pool_resource<Upstream, Bookkeeper>(upstream, bookkeeper);
}();

return adaptor;
}



} 
} 

#endif 

