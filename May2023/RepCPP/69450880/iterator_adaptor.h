






#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>
#include <hydra/detail/external/hydra_thrust/detail/use_default.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_adaptor_base.h>

namespace hydra_thrust
{






template<typename Derived,
typename Base,
typename Value      = use_default,
typename System     = use_default,
typename Traversal  = use_default,
typename Reference  = use_default,
typename Difference = use_default>
class iterator_adaptor:
public detail::iterator_adaptor_base<
Derived, Base, Value, System, Traversal, Reference, Difference
>::type
{


friend class hydra_thrust::iterator_core_access;

protected:
typedef typename detail::iterator_adaptor_base<
Derived, Base, Value, System, Traversal, Reference, Difference
>::type super_t;



public:

__host__ __device__
iterator_adaptor(){}


__hydra_thrust_exec_check_disable__
__host__ __device__
explicit iterator_adaptor(Base const& iter)
: m_iterator(iter)
{}


typedef Base       base_type;


typedef typename super_t::reference reference;

typedef typename super_t::difference_type difference_type;



__host__ __device__
Base const& base() const
{ return m_iterator; }

protected:

__host__ __device__
Base const& base_reference() const
{ return m_iterator; }


__host__ __device__
Base& base_reference()
{ return m_iterator; }


private: 

__hydra_thrust_exec_check_disable__
__host__ __device__
typename iterator_adaptor::reference dereference() const
{ return *m_iterator; }

__hydra_thrust_exec_check_disable__
template<typename OtherDerived, typename OtherIterator, typename V, typename S, typename T, typename R, typename D>
__host__ __device__
bool equal(iterator_adaptor<OtherDerived, OtherIterator, V, S, T, R, D> const& x) const
{ return m_iterator == x.base(); }

__hydra_thrust_exec_check_disable__
__host__ __device__
void advance(typename iterator_adaptor::difference_type n)
{
m_iterator += n;
}

__hydra_thrust_exec_check_disable__
__host__ __device__
void increment()
{ ++m_iterator; }

__hydra_thrust_exec_check_disable__
__host__ __device__
void decrement()
{
--m_iterator;
}

__hydra_thrust_exec_check_disable__
template<typename OtherDerived, typename OtherIterator, typename V, typename S, typename T, typename R, typename D>
__host__ __device__
typename iterator_adaptor::difference_type distance_to(iterator_adaptor<OtherDerived, OtherIterator, V, S, T, R, D> const& y) const
{ return y.base() - m_iterator; }

private:
Base m_iterator;


}; 





} 

