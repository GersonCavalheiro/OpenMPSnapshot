

#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/tuple/tuple_transform.h>


namespace hydra_thrust
{


template<typename IteratorTuple>
__host__ __device__
zip_iterator<IteratorTuple>
::zip_iterator(void)
{
} 


template<typename IteratorTuple>
__host__ __device__
zip_iterator<IteratorTuple>
::zip_iterator(IteratorTuple iterator_tuple)
:m_iterator_tuple(iterator_tuple)
{
} 


template<typename IteratorTuple>
template<typename OtherIteratorTuple>
__host__ __device__
zip_iterator<IteratorTuple>
::zip_iterator(const zip_iterator<OtherIteratorTuple> &other,
typename hydra_thrust::detail::enable_if_convertible<
OtherIteratorTuple,
IteratorTuple
>::type *)
:m_iterator_tuple(other.get_iterator_tuple())
{
} 


template<typename IteratorTuple>
__host__ __device__
const IteratorTuple &zip_iterator<IteratorTuple>
::get_iterator_tuple(void) const
{
return m_iterator_tuple;
} 


template<typename IteratorTuple>
typename zip_iterator<IteratorTuple>::super_t::reference
__host__ __device__
zip_iterator<IteratorTuple>
::dereference(void) const
{
using namespace detail::tuple_impl_specific;

return hydra_thrust::detail::tuple_host_device_transform<detail::dereference_iterator::template apply>(get_iterator_tuple(), detail::dereference_iterator());
} 


__hydra_thrust_exec_check_disable__
template<typename IteratorTuple>
template<typename OtherIteratorTuple>
__host__ __device__
bool zip_iterator<IteratorTuple>
::equal(const zip_iterator<OtherIteratorTuple> &other) const
{
return hydra_thrust::get<0>(get_iterator_tuple()) == hydra_thrust::get<0>(other.get_iterator_tuple());
} 


template<typename IteratorTuple>
__host__ __device__
void zip_iterator<IteratorTuple>
::advance(typename super_t::difference_type n)
{
using namespace detail::tuple_impl_specific;
tuple_for_each(m_iterator_tuple,
detail::advance_iterator<typename super_t::difference_type>(n));
} 


template<typename IteratorTuple>
__host__ __device__
void zip_iterator<IteratorTuple>
::increment(void)
{
using namespace detail::tuple_impl_specific;
tuple_for_each(m_iterator_tuple, detail::increment_iterator());
} 


template<typename IteratorTuple>
__host__ __device__
void zip_iterator<IteratorTuple>
::decrement(void)
{
using namespace detail::tuple_impl_specific;
tuple_for_each(m_iterator_tuple, detail::decrement_iterator());
} 


__hydra_thrust_exec_check_disable__
template<typename IteratorTuple>
template <typename OtherIteratorTuple>
__host__ __device__
typename zip_iterator<IteratorTuple>::super_t::difference_type
zip_iterator<IteratorTuple>
::distance_to(const zip_iterator<OtherIteratorTuple> &other) const
{
return hydra_thrust::get<0>(other.get_iterator_tuple()) - hydra_thrust::get<0>(get_iterator_tuple());
} 

#ifdef HYDRA_THRUST_VARIADIC_TUPLE
template<typename... Iterators>
__host__ __device__
zip_iterator<hydra_thrust::tuple<Iterators...>> make_zip_iterator(hydra_thrust::tuple<Iterators...> t)
{
return zip_iterator<hydra_thrust::tuple<Iterators...>>(t);
} 


template<typename... Iterators>
__host__ __device__
zip_iterator<hydra_thrust::tuple<Iterators...>> make_zip_iterator(Iterators... its)
{
return make_zip_iterator(hydra_thrust::make_tuple(its...));
} 
#else
template<typename IteratorTuple>
__host__ __device__
zip_iterator<IteratorTuple> make_zip_iterator(IteratorTuple t)
{
return zip_iterator<IteratorTuple>(t);
} 
#endif

} 

