

#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/numeric_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <cstddef>

namespace hydra_thrust
{

template <typename Incrementable, typename System, typename Traversal, typename Difference>
class counting_iterator;

namespace detail
{

template <typename Incrementable, typename System, typename Traversal, typename Difference>
struct counting_iterator_base
{
typedef typename hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_same<System,use_default>::value,
hydra_thrust::detail::identity_<hydra_thrust::any_system_tag>,
hydra_thrust::detail::identity_<System>
>::type system;

typedef typename hydra_thrust::detail::ia_dflt_help<
Traversal,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_numeric<Incrementable>::value,
hydra_thrust::detail::identity_<random_access_traversal_tag>,
hydra_thrust::iterator_traversal<Incrementable>
>
>::type traversal;

typedef typename hydra_thrust::detail::ia_dflt_help<
Difference,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_numeric<Incrementable>::value,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_integral<Incrementable>::value,
hydra_thrust::detail::numeric_difference<Incrementable>,
hydra_thrust::detail::identity_<std::ptrdiff_t>
>,
hydra_thrust::iterator_difference<Incrementable>
>
>::type difference;

typedef hydra_thrust::iterator_adaptor<
counting_iterator<Incrementable, System, Traversal, Difference>, 
Incrementable,                                                  
Incrementable,                                                  
system,
traversal,
Incrementable,
difference
> type;
}; 


template<typename Difference, typename Incrementable1, typename Incrementable2>
struct iterator_distance
{
__host__ __device__
static Difference distance(Incrementable1 x, Incrementable2 y)
{
return y - x;
}
};


template<typename Difference, typename Incrementable1, typename Incrementable2>
struct number_distance
{
__host__ __device__
static Difference distance(Incrementable1 x, Incrementable2 y)
{
return static_cast<Difference>(numeric_distance(x,y));
}
};


template<typename Difference, typename Incrementable1, typename Incrementable2, typename Enable = void>
struct counting_iterator_equal
{
__host__ __device__
static bool equal(Incrementable1 x, Incrementable2 y)
{
return x == y;
}
};


template<typename Difference, typename Incrementable1, typename Incrementable2>
struct counting_iterator_equal<
Difference,
Incrementable1,
Incrementable2,
typename hydra_thrust::detail::enable_if<
hydra_thrust::detail::is_floating_point<Incrementable1>::value ||
hydra_thrust::detail::is_floating_point<Incrementable2>::value
>::type
>
{
__host__ __device__
static bool equal(Incrementable1 x, Incrementable2 y)
{
typedef number_distance<Difference,Incrementable1,Incrementable2> d;
return d::distance(x,y) == 0;
}
};


} 
} 

