



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/mr/pool.h>

namespace hydra_thrust
{
namespace mr
{




template<typename Upstream, typename Bookkeeper>
__host__ __device__
hydra_thrust::mr::unsynchronized_pool_resource<Upstream> & tls_pool(Upstream * upstream = NULL)
{
static thread_local auto adaptor = [&]{
assert(upstream);
return hydra_thrust::mr::unsynchronized_pool_resource<Upstream>(upstream);
}();

return adaptor;
}



} 
} 

#endif 

