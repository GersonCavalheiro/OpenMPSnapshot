






#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <hydra/detail/external/hydra_thrust/iterator/detail/transform_iterator.inl>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{






template <class AdaptableUnaryFunction, class Iterator, class Reference = use_default, class Value = use_default>
class transform_iterator
: public detail::transform_iterator_base<AdaptableUnaryFunction, Iterator, Reference, Value>::type
{

public:
typedef typename
detail::transform_iterator_base<AdaptableUnaryFunction, Iterator, Reference, Value>::type
super_t;

friend class hydra_thrust::iterator_core_access;


public:

__host__ __device__
transform_iterator() {}


__host__ __device__
transform_iterator(Iterator const& x, AdaptableUnaryFunction f)
: super_t(x), m_f(f) {
}


__host__ __device__
explicit transform_iterator(Iterator const& x)
: super_t(x) { }


template<typename OtherAdaptableUnaryFunction,
typename OtherIterator,
typename OtherReference,
typename OtherValue>
__host__ __device__
transform_iterator(const transform_iterator<OtherAdaptableUnaryFunction, OtherIterator, OtherReference, OtherValue> &other,
typename hydra_thrust::detail::enable_if_convertible<OtherIterator, Iterator>::type* = 0,
typename hydra_thrust::detail::enable_if_convertible<OtherAdaptableUnaryFunction, AdaptableUnaryFunction>::type* = 0)
: super_t(other.base()), m_f(other.functor()) {}


__host__ __device__
transform_iterator &operator=(const transform_iterator &other)
{
return do_assign(other,
#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC) && (HYDRA_THRUST_GCC_VERSION <= 40201)
hydra_thrust::detail::true_type()
#else
typename hydra_thrust::detail::is_copy_assignable<AdaptableUnaryFunction>::type()
#endif 
);
}


__host__ __device__
AdaptableUnaryFunction functor() const
{ return m_f; }


private:
__host__ __device__
transform_iterator &do_assign(const transform_iterator &other, hydra_thrust::detail::true_type)
{
super_t::operator=(other);

m_f = other.functor();

return *this;
}

__host__ __device__
transform_iterator &do_assign(const transform_iterator &other, hydra_thrust::detail::false_type)
{
super_t::operator=(other);


return *this;
}

HYDRA_THRUST_DISABLE_MSVC_WARNING_BEGIN(4172)

__hydra_thrust_exec_check_disable__
__host__ __device__
typename super_t::reference dereference() const
{  
typename hydra_thrust::iterator_value<Iterator>::type x = *this->base();
return m_f(x);
}

HYDRA_THRUST_DISABLE_MSVC_WARNING_END(4172)

mutable AdaptableUnaryFunction m_f;


}; 



template <class AdaptableUnaryFunction, class Iterator>
inline __host__ __device__
transform_iterator<AdaptableUnaryFunction, Iterator>
make_transform_iterator(Iterator it, AdaptableUnaryFunction fun)
{
return transform_iterator<AdaptableUnaryFunction, Iterator>(it, fun);
} 





} 

