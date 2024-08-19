




#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/type_traits/is_contiguous_iterator.h>

namespace hydra_thrust
{
namespace detail
{


template<typename Pointer>
class normal_iterator
: public iterator_adaptor<
normal_iterator<Pointer>,
Pointer
>
{
typedef iterator_adaptor<normal_iterator<Pointer>, Pointer> super_t;

public:
__host__ __device__
normal_iterator() {}

__host__ __device__
normal_iterator(Pointer p)
: super_t(p) {}

template<typename OtherPointer>
__host__ __device__
normal_iterator(const normal_iterator<OtherPointer> &other,
typename hydra_thrust::detail::enable_if_convertible<
OtherPointer,
Pointer
>::type * = 0)
: super_t(other.base()) {}

}; 


template<typename Pointer>
inline __host__ __device__ normal_iterator<Pointer> make_normal_iterator(Pointer ptr)
{
return normal_iterator<Pointer>(ptr);
}

} 

template <typename T>
struct proclaim_contiguous_iterator<
hydra_thrust::detail::normal_iterator<T>
> : true_type {};

} 

