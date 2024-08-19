


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/tuple.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<unsigned int i, typename Env>
struct argument_helper
{
typedef typename hydra_thrust::tuple_element<i,Env>::type type;
};

template<unsigned int i>
struct argument_helper<i,hydra_thrust::null_type>
{
typedef hydra_thrust::null_type type;
};


template<unsigned int i>
class argument
{
public:
template<typename Env>
struct result
: argument_helper<i,Env>
{
};

__host__ __device__
argument(void){}

template<typename Env>
__host__ __device__
typename result<Env>::type eval(const Env &e) const
{
return hydra_thrust::get<i>(e);
} 
}; 

} 
} 
} 

