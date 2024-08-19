

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_reference_cast.h>

namespace hydra_thrust
{
namespace detail
{


template<typename Function, typename Result>
struct wrapped_function
{
mutable Function m_f;

inline __host__ __device__
wrapped_function()
: m_f()
{}

inline __host__ __device__
wrapped_function(const Function &f)
: m_f(f)
{}

__hydra_thrust_exec_check_disable__
template<typename Argument>
inline __host__ __device__
Result operator()(Argument &x) const
{
return static_cast<Result>(m_f(hydra_thrust::raw_reference_cast(x)));
}

__hydra_thrust_exec_check_disable__
template<typename Argument>
inline __host__ __device__ Result operator()(const Argument &x) const
{
return static_cast<Result>(m_f(hydra_thrust::raw_reference_cast(x)));
}

__hydra_thrust_exec_check_disable__
template<typename Argument1, typename Argument2>
inline __host__ __device__ Result operator()(Argument1 &x, Argument2 &y) const
{
return static_cast<Result>(m_f(hydra_thrust::raw_reference_cast(x), hydra_thrust::raw_reference_cast(y)));
}

__hydra_thrust_exec_check_disable__
template<typename Argument1, typename Argument2>
inline __host__ __device__ Result operator()(const Argument1 &x, Argument2 &y) const
{
return static_cast<Result>(m_f(hydra_thrust::raw_reference_cast(x), hydra_thrust::raw_reference_cast(y)));
}

__hydra_thrust_exec_check_disable__
template<typename Argument1, typename Argument2>
inline __host__ __device__ Result operator()(const Argument1 &x, const Argument2 &y) const
{
return static_cast<Result>(m_f(hydra_thrust::raw_reference_cast(x), hydra_thrust::raw_reference_cast(y)));
}

__hydra_thrust_exec_check_disable__
template<typename Argument1, typename Argument2>
inline __host__ __device__ Result operator()(Argument1 &x, const Argument2 &y) const
{
return static_cast<Result>(m_f(hydra_thrust::raw_reference_cast(x), hydra_thrust::raw_reference_cast(y)));
}
}; 


} 
} 

