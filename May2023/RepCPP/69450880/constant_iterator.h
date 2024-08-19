




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/constant_iterator_base.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>

namespace hydra_thrust
{






template<typename Value,
typename Incrementable = use_default,
typename System = use_default>
class constant_iterator
: public detail::constant_iterator_base<Value, Incrementable, System>::type
{

friend class hydra_thrust::iterator_core_access;
typedef typename detail::constant_iterator_base<Value, Incrementable, System>::type          super_t;
typedef typename detail::constant_iterator_base<Value, Incrementable, System>::incrementable incrementable;
typedef typename detail::constant_iterator_base<Value, Incrementable, System>::base_iterator base_iterator;

public:
typedef typename super_t::reference  reference;
typedef typename super_t::value_type value_type;




__host__ __device__
constant_iterator()
: super_t(), m_value() {}


__host__ __device__
constant_iterator(constant_iterator const &rhs)
: super_t(rhs.base()), m_value(rhs.m_value) {}


template<typename OtherSystem>
__host__ __device__
constant_iterator(constant_iterator<Value,Incrementable,OtherSystem> const &rhs,
typename hydra_thrust::detail::enable_if_convertible<
typename hydra_thrust::iterator_system<constant_iterator<Value,Incrementable,OtherSystem> >::type,
typename hydra_thrust::iterator_system<super_t>::type
>::type * = 0)
: super_t(rhs.base()), m_value(rhs.value()) {}


__host__ __device__
constant_iterator(value_type const& v, incrementable const &i = incrementable())
: super_t(base_iterator(i)), m_value(v) {}


template<typename OtherValue, typename OtherIncrementable>
__host__ __device__
constant_iterator(OtherValue const& v, OtherIncrementable const& i = incrementable())
: super_t(base_iterator(i)), m_value(v) {}


__host__ __device__
Value const& value() const
{ return m_value; }



protected:
__host__ __device__
Value const& value_reference() const
{ return m_value; }

__host__ __device__
Value & value_reference()
{ return m_value; }

private: 
__host__ __device__
reference dereference() const
{
return m_value;
}

private:
Value m_value;


}; 



template<typename V, typename I>
inline __host__ __device__
constant_iterator<V,I> make_constant_iterator(V x, I i = int())
{
return constant_iterator<V,I>(x, i);
} 



template<typename V>
inline __host__ __device__
constant_iterator<V> make_constant_iterator(V x)
{
return constant_iterator<V>(x, 0);
} 





} 

