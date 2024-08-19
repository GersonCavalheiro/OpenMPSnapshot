




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/reference.h>

namespace hydra_thrust
{




template<typename T>
class device_reference
: public hydra_thrust::reference<
T,
hydra_thrust::device_ptr<T>,
hydra_thrust::device_reference<T>
>
{
private:
typedef hydra_thrust::reference<
T,
hydra_thrust::device_ptr<T>,
hydra_thrust::device_reference<T>
> super_t;

public:

typedef typename super_t::value_type value_type;


typedef typename super_t::pointer    pointer;


template<typename OtherT>
__host__ __device__
device_reference(const device_reference<OtherT> &other,
typename hydra_thrust::detail::enable_if_convertible<
typename device_reference<OtherT>::pointer,
pointer
>::type * = 0)
: super_t(other)
{}


__host__ __device__
explicit device_reference(const pointer &ptr)
: super_t(ptr)
{}


template<typename OtherT>
__host__ __device__
device_reference &operator=(const device_reference<OtherT> &other);


__host__ __device__
device_reference &operator=(const value_type &x);

#if 0

__host__ __device__
pointer operator&(void) const;


__host__ __device__
operator value_type (void) const;


__host__ __device__
void swap(device_reference &other);


device_reference &operator++(void);


value_type operator++(int);


device_reference &operator+=(const T &rhs);


device_reference &operator--(void);


value_type operator--(int);


device_reference &operator-=(const T &rhs);


device_reference &operator*=(const T &rhs);


device_reference &operator/=(const T &rhs);


device_reference &operator%=(const T &rhs);


device_reference &operator<<=(const T &rhs);


device_reference &operator>>=(const T &rhs);


device_reference &operator&=(const T &rhs);


device_reference &operator|=(const T &rhs);


device_reference &operator^=(const T &rhs);
#endif 
}; 


template<typename T>
__host__ __device__
void swap(device_reference<T> x, device_reference<T> y);

#if 0

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os, const device_reference<T> &y);
#endif



} 

#include <hydra/detail/external/hydra_thrust/detail/device_reference.inl>

