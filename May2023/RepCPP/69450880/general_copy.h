



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_reference_cast.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace sequential
{
namespace general_copy_detail
{


template<typename T1, typename T2>
struct lazy_is_assignable
: hydra_thrust::detail::is_assignable<
typename T1::type,
typename T2::type
>
{};


template<typename InputIterator, typename OutputIterator>
struct reference_is_assignable
: hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_same<
typename hydra_thrust::iterator_reference<OutputIterator>::type, void
>::value,
hydra_thrust::detail::true_type,
lazy_is_assignable<
hydra_thrust::iterator_reference<OutputIterator>,
hydra_thrust::iterator_reference<InputIterator>
>
>::type
{};



__hydra_thrust_exec_check_disable__
template<typename OutputIterator, typename InputIterator>
inline __host__ __device__
typename hydra_thrust::detail::enable_if<
reference_is_assignable<InputIterator,OutputIterator>::value
>::type
iter_assign(OutputIterator dst, InputIterator src)
{
*dst = *src;
}


__hydra_thrust_exec_check_disable__
template<typename OutputIterator, typename InputIterator>
inline __host__ __device__
typename hydra_thrust::detail::disable_if<
reference_is_assignable<InputIterator,OutputIterator>::value
>::type
iter_assign(OutputIterator dst, InputIterator src)
{
typedef typename hydra_thrust::iterator_value<InputIterator>::type value_type;

*dst = static_cast<value_type>(*src);
}


} 


__hydra_thrust_exec_check_disable__
template<typename InputIterator,
typename OutputIterator>
__host__ __device__
OutputIterator general_copy(InputIterator first,
InputIterator last,
OutputIterator result)
{
for(; first != last; ++first, ++result)
{
#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC) && (HYDRA_THRUST_GCC_VERSION < 40300)
*result = *first;
#else
general_copy_detail::iter_assign(result, first);
#endif
}

return result;
} 


__hydra_thrust_exec_check_disable__
template<typename InputIterator,
typename Size,
typename OutputIterator>
__host__ __device__
OutputIterator general_copy_n(InputIterator first,
Size n,
OutputIterator result)
{
for(; n > Size(0); ++first, ++result, --n)
{
#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC) && (HYDRA_THRUST_GCC_VERSION < 40300)
*result = *first;
#else
general_copy_detail::iter_assign(result, first);
#endif
}

return result;
} 


} 
} 
} 
} 

