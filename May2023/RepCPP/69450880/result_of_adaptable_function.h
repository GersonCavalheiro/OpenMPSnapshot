

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/function_traits.h>

#if __cplusplus >= 201103L || defined(__cpp_lib_result_of_sfinae)
#include <type_traits>
#endif

namespace hydra_thrust
{
namespace detail
{

#if __cplusplus >= 201103L || defined(__cpp_lib_result_of_sfinae)
template <typename Signature, typename Enable = void>
struct result_of_adaptable_function : std::result_of<Signature> {};
#else  
template<typename Signature, typename Enable = void> 
struct result_of_adaptable_function;
#endif  

template<typename Functor, typename Arg1>
struct result_of_adaptable_function<
Functor(Arg1),
typename hydra_thrust::detail::enable_if<hydra_thrust::detail::has_result_type<Functor>::value>::type
>
{
typedef typename Functor::result_type type;
}; 

template<typename Functor, typename Arg1, typename Arg2>
struct result_of_adaptable_function<
Functor(Arg1,Arg2),
typename hydra_thrust::detail::enable_if<hydra_thrust::detail::has_result_type<Functor>::value>::type
>
{
typedef typename Functor::result_type type;
};


} 
} 

