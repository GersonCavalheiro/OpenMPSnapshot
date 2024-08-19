
#pragma once

#include <hydra/detail/external/hydra_thrust/detail/seq.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/par.h>

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {

template <int PAR>
struct has_par : hydra_thrust::detail::true_type {};

template <>
struct has_par<0> : hydra_thrust::detail::false_type {};

template<class Policy>
struct cvt_to_seq_impl
{
typedef hydra_thrust::detail::seq_t seq_t;

static seq_t __host__ __device__
doit(Policy&)
{
return seq_t();
}
};    

#if 0
template <class Allocator>
struct cvt_to_seq_impl<
hydra_thrust::detail::execute_with_allocator<Allocator,
execute_on_stream_base> >
{
typedef hydra_thrust::detail::execute_with_allocator<Allocator,
execute_on_stream_base>
Policy;
typedef hydra_thrust::detail::execute_with_allocator<
Allocator,
hydra_thrust::system::detail::sequential::execution_policy>
seq_t;


static seq_t __host__ __device__
doit(Policy& policy)
{
return seq_t(policy.m_alloc);
}
};    
#endif

template <class Policy>
typename cvt_to_seq_impl<Policy>::seq_t __host__ __device__
cvt_to_seq(Policy& policy)
{
return cvt_to_seq_impl<Policy>::doit(policy);
}

#if __HYDRA_THRUST_HAS_CUDART__
#define HYDRA_THRUST_CUDART_DISPATCH par
#else
#define HYDRA_THRUST_CUDART_DISPATCH seq
#endif

} 
HYDRA_THRUST_END_NS
