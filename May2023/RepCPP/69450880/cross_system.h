
#pragma once

#include <hydra/detail/external/hydra_thrust/system/cuda/detail/guarded_cuda_runtime_api.h>
#include <hydra/detail/external/hydra_thrust/system/cpp/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/execution_policy.h>

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {

template <class Sys1, class Sys2>
struct cross_system : execution_policy<cross_system<Sys1, Sys2> >
{
typedef hydra_thrust::execution_policy<Sys1> policy1;
typedef hydra_thrust::execution_policy<Sys2> policy2;

policy1 &sys1;
policy2 &sys2;

inline __host__ __device__
cross_system(policy1 &sys1, policy2 &sys2) : sys1(sys1), sys2(sys2) {}

inline __host__ __device__
cross_system<Sys2, Sys1> rotate() const
{
return cross_system<Sys2, Sys1>(sys2, sys1);
}
};

#if HYDRA_THRUST_CPP_DIALECT >= 2011
template <class Sys1, class Sys2>
HYDRA_THRUST_CONSTEXPR __host__ __device__ 
auto direction_of_copy(
hydra_thrust::system::cuda::execution_policy<Sys1> const&
, hydra_thrust::cpp::execution_policy<Sys2> const&
)
HYDRA_THRUST_DECLTYPE_RETURNS(
hydra_thrust::detail::integral_constant<
cudaMemcpyKind, cudaMemcpyDeviceToHost
>{}
)

template <class Sys1, class Sys2>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto direction_of_copy(
hydra_thrust::cpp::execution_policy<Sys1> const&
, hydra_thrust::system::cuda::execution_policy<Sys2> const&
)
HYDRA_THRUST_DECLTYPE_RETURNS(
hydra_thrust::detail::integral_constant<
cudaMemcpyKind, cudaMemcpyHostToDevice
>{}
)

template <class Sys1, class Sys2>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto direction_of_copy(
hydra_thrust::system::cuda::execution_policy<Sys1> const&
, hydra_thrust::system::cuda::execution_policy<Sys2> const&
)
HYDRA_THRUST_DECLTYPE_RETURNS(
hydra_thrust::detail::integral_constant<
cudaMemcpyKind, cudaMemcpyDeviceToDevice
>{}
)

template <class DerivedPolicy>
HYDRA_THRUST_CONSTEXPR __host__ __device__ 
auto direction_of_copy(execution_policy<DerivedPolicy> const &)
HYDRA_THRUST_DECLTYPE_RETURNS(
hydra_thrust::detail::integral_constant<
cudaMemcpyKind, cudaMemcpyDeviceToDevice
>{}
)

template <class Sys1, class Sys2>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto direction_of_copy(
execution_policy<cross_system<Sys1, Sys2>> const &systems
)
HYDRA_THRUST_DECLTYPE_RETURNS(
direction_of_copy(
derived_cast(derived_cast(systems).sys1)
, derived_cast(derived_cast(systems).sys2)
)
)

template <typename ExecutionPolicy0, typename ExecutionPolicy1>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto is_device_to_host_copy(
ExecutionPolicy0 const& exec0
, ExecutionPolicy1 const& exec1
)
noexcept -> 
hydra_thrust::detail::integral_constant<
bool
,    cudaMemcpyDeviceToHost
== decltype(direction_of_copy(exec0, exec1))::value
>
{
return {};
}

template <typename ExecutionPolicy>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto is_device_to_host_copy(ExecutionPolicy const& exec)
noexcept -> 
hydra_thrust::detail::integral_constant<
bool
,    cudaMemcpyDeviceToHost
== decltype(direction_of_copy(exec))::value
>
{
return {};
}

template <typename ExecutionPolicy0, typename ExecutionPolicy1>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto is_host_to_device_copy(
ExecutionPolicy0 const& exec0
, ExecutionPolicy1 const& exec1
)
noexcept -> 
hydra_thrust::detail::integral_constant<
bool
,    cudaMemcpyHostToDevice
== decltype(direction_of_copy(exec0, exec1))::value
>
{
return {};
}

template <typename ExecutionPolicy>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto is_host_to_device_copy(ExecutionPolicy const& exec)
noexcept -> 
hydra_thrust::detail::integral_constant<
bool
,    cudaMemcpyHostToDevice
== decltype(direction_of_copy(exec))::value
>
{
return {};
}

template <typename ExecutionPolicy0, typename ExecutionPolicy1>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto is_device_to_device_copy(
ExecutionPolicy0 const& exec0
, ExecutionPolicy1 const& exec1
)
noexcept -> 
hydra_thrust::detail::integral_constant<
bool
,    cudaMemcpyDeviceToDevice
== decltype(direction_of_copy(exec0, exec1))::value
>
{
return {};
}

template <typename ExecutionPolicy>
HYDRA_THRUST_CONSTEXPR __host__ __device__
auto is_device_to_device_copy(ExecutionPolicy const& exec)
noexcept -> 
hydra_thrust::detail::integral_constant<
bool
,    cudaMemcpyDeviceToDevice
== decltype(direction_of_copy(exec))::value
>
{
return {};
}


template <class Sys1, class Sys2>
__host__ __device__
auto
select_device_system(hydra_thrust::cuda::execution_policy<Sys1> &sys1,
hydra_thrust::execution_policy<Sys2> &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_device_system(hydra_thrust::cuda::execution_policy<Sys1> const &sys1,
hydra_thrust::execution_policy<Sys2> const &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_device_system(hydra_thrust::execution_policy<Sys1> &,
hydra_thrust::cuda::execution_policy<Sys2> &sys2)
HYDRA_THRUST_DECLTYPE_RETURNS(sys2)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_device_system(hydra_thrust::execution_policy<Sys1> const &,
hydra_thrust::cuda::execution_policy<Sys2> const &sys2)
HYDRA_THRUST_DECLTYPE_RETURNS(sys2)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_device_system(hydra_thrust::cuda::execution_policy<Sys1> &sys1,
hydra_thrust::cuda::execution_policy<Sys2> &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_device_system(hydra_thrust::cuda::execution_policy<Sys1> const &sys1,
hydra_thrust::cuda::execution_policy<Sys2> const &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)


template <class Sys1, class Sys2>
__host__ __device__
auto
select_host_system(hydra_thrust::cuda::execution_policy<Sys1> &,
hydra_thrust::execution_policy<Sys2> &sys2)
HYDRA_THRUST_DECLTYPE_RETURNS(sys2)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_host_system(hydra_thrust::cuda::execution_policy<Sys1> const &,
hydra_thrust::execution_policy<Sys2> const &sys2)
HYDRA_THRUST_DECLTYPE_RETURNS(sys2)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_host_system(hydra_thrust::execution_policy<Sys1> &sys1,
hydra_thrust::cuda::execution_policy<Sys2> &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_host_system(hydra_thrust::execution_policy<Sys1> const &sys1,
hydra_thrust::cuda::execution_policy<Sys2> const &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_host_system(hydra_thrust::execution_policy<Sys1> &sys1,
hydra_thrust::execution_policy<Sys2> &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)

template <class Sys1, class Sys2>
__host__ __device__
auto
select_host_system(hydra_thrust::execution_policy<Sys1> const &sys1,
hydra_thrust::execution_policy<Sys2> const &)
HYDRA_THRUST_DECLTYPE_RETURNS(sys1)
#endif

template <class Sys1, class Sys2>
__host__ __device__
cross_system<Sys1, Sys2>
select_system(execution_policy<Sys1> const &             sys1,
hydra_thrust::cpp::execution_policy<Sys2> const &sys2)
{
hydra_thrust::execution_policy<Sys1> &     non_const_sys1 = const_cast<execution_policy<Sys1> &>(sys1);
hydra_thrust::cpp::execution_policy<Sys2> &non_const_sys2 = const_cast<hydra_thrust::cpp::execution_policy<Sys2> &>(sys2);
return cross_system<Sys1, Sys2>(non_const_sys1, non_const_sys2);
}

template <class Sys1, class Sys2>
__host__ __device__
cross_system<Sys1, Sys2>
select_system(hydra_thrust::cpp::execution_policy<Sys1> const &sys1,
execution_policy<Sys2> const &             sys2)
{
hydra_thrust::cpp::execution_policy<Sys1> &non_const_sys1 = const_cast<hydra_thrust::cpp::execution_policy<Sys1> &>(sys1);
hydra_thrust::execution_policy<Sys2> &     non_const_sys2 = const_cast<execution_policy<Sys2> &>(sys2);
return cross_system<Sys1, Sys2>(non_const_sys1, non_const_sys2);
}

} 
HYDRA_THRUST_END_NS

