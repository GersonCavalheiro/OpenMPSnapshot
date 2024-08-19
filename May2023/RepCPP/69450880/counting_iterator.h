






#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_categories.h>

#include <hydra/detail/external/hydra_thrust/iterator/detail/counting_iterator.inl>

namespace hydra_thrust
{






template<typename Incrementable,
typename System = use_default,
typename Traversal = use_default,
typename Difference = use_default>
class counting_iterator
: public detail::counting_iterator_base<Incrementable, System, Traversal, Difference>::type
{

typedef typename detail::counting_iterator_base<Incrementable, System, Traversal, Difference>::type super_t;

friend class hydra_thrust::iterator_core_access;

public:
typedef typename super_t::reference       reference;
typedef typename super_t::difference_type difference_type;




__host__ __device__
counting_iterator() {}


__host__ __device__
counting_iterator(counting_iterator const &rhs):super_t(rhs.base()){}


template<typename OtherSystem>
__host__ __device__
counting_iterator(counting_iterator<Incrementable, OtherSystem, Traversal, Difference> const &rhs,
typename hydra_thrust::detail::enable_if_convertible<
typename hydra_thrust::iterator_system<counting_iterator<Incrementable,OtherSystem,Traversal,Difference> >::type,
typename hydra_thrust::iterator_system<super_t>::type
>::type * = 0)
: super_t(rhs.base()){}


__host__ __device__
explicit counting_iterator(Incrementable x):super_t(x){}


private:
__host__ __device__
reference dereference() const
{
return this->base_reference();
}

template <typename OtherIncrementable, typename OtherSystem, typename OtherTraversal, typename OtherDifference>
__host__ __device__
bool equal(counting_iterator<OtherIncrementable, OtherSystem, OtherTraversal, OtherDifference> const& y) const
{
typedef hydra_thrust::detail::counting_iterator_equal<difference_type,Incrementable,OtherIncrementable> e;
return e::equal(this->base(), y.base());
}

template <class OtherIncrementable>
__host__ __device__
difference_type
distance_to(counting_iterator<OtherIncrementable, System, Traversal, Difference> const& y) const
{
typedef typename
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_numeric<Incrementable>::value,
hydra_thrust::detail::identity_<hydra_thrust::detail::number_distance<difference_type, Incrementable, OtherIncrementable> >,
hydra_thrust::detail::identity_<hydra_thrust::detail::iterator_distance<difference_type, Incrementable, OtherIncrementable> >
>::type d;

return d::distance(this->base(), y.base());
}


}; 



template <typename Incrementable>
inline __host__ __device__
counting_iterator<Incrementable> make_counting_iterator(Incrementable x)
{
return counting_iterator<Incrementable>(x);
}





} 

