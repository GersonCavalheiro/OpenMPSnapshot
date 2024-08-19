






#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/zip_iterator_base.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{






template <typename IteratorTuple>
class zip_iterator
: public detail::zip_iterator_base<IteratorTuple>::type
{
public:

inline __host__ __device__
zip_iterator(void);


inline __host__ __device__
zip_iterator(IteratorTuple iterator_tuple);


template<typename OtherIteratorTuple>
inline __host__ __device__
zip_iterator(const zip_iterator<OtherIteratorTuple> &other,
typename hydra_thrust::detail::enable_if_convertible<
OtherIteratorTuple,
IteratorTuple
>::type * = 0);


inline __host__ __device__
const IteratorTuple &get_iterator_tuple() const;


private:
typedef typename
detail::zip_iterator_base<IteratorTuple>::type super_t;

friend class hydra_thrust::iterator_core_access;

__host__ __device__
typename super_t::reference dereference() const;

template<typename OtherIteratorTuple>
inline __host__ __device__
bool equal(const zip_iterator<OtherIteratorTuple> &other) const;

inline __host__ __device__
void advance(typename super_t::difference_type n);

inline __host__ __device__
void increment();

inline __host__ __device__
void decrement();

template<typename OtherIteratorTuple>
inline __host__ __device__
typename super_t::difference_type
distance_to(const zip_iterator<OtherIteratorTuple> &other) const;

IteratorTuple m_iterator_tuple;


}; 


#ifdef HYDRA_THRUST_VARIADIC_TUPLE
template<typename... Iterators>
inline __host__ __device__
zip_iterator<hydra_thrust::tuple<Iterators...>> make_zip_iterator(hydra_thrust::tuple<Iterators...> t);



template<typename... Iterators>
inline __host__ __device__
zip_iterator<hydra_thrust::tuple<Iterators...>> make_zip_iterator(Iterators... its);
#else
template<typename IteratorTuple>
inline __host__ __device__
zip_iterator<IteratorTuple> make_zip_iterator(IteratorTuple t);
#endif






} 

#include <hydra/detail/external/hydra_thrust/iterator/detail/zip_iterator.inl>

