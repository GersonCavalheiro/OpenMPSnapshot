




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/memory.h>

namespace hydra_thrust
{



template<typename T> class device_reference;


template<typename T>
class device_ptr
: public hydra_thrust::pointer<
T,
hydra_thrust::device_system_tag,
hydra_thrust::device_reference<T>,
hydra_thrust::device_ptr<T>
>
{
private:
typedef hydra_thrust::pointer<
T,
hydra_thrust::device_system_tag,
hydra_thrust::device_reference<T>,
hydra_thrust::device_ptr<T>
> super_t;

public:

__host__ __device__
device_ptr() : super_t() {}

#if HYDRA_THRUST_CPP_DIALECT >= 2011
__host__ __device__
device_ptr(decltype(nullptr)) : super_t(nullptr) {}
#endif


template<typename OtherT>
__host__ __device__
explicit device_ptr(OtherT *ptr) : super_t(ptr) {}


template<typename OtherT>
__host__ __device__
device_ptr(const device_ptr<OtherT> &other) : super_t(other) {}


template<typename OtherT>
__host__ __device__
device_ptr &operator=(const device_ptr<OtherT> &other)
{
super_t::operator=(other);
return *this;
}

#if HYDRA_THRUST_CPP_DIALECT >= 2011
__host__ __device__
device_ptr& operator=(decltype(nullptr))
{
super_t::operator=(nullptr);
return *this;
}
#endif

#if 0

__host__ __device__
T *get(void) const;
#endif 
}; 

#if 0

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os, const device_ptr<T> &p);
#endif







template<typename T>
__host__ __device__
inline device_ptr<T> device_pointer_cast(T *ptr);


template<typename T>
__host__ __device__
inline device_ptr<T> device_pointer_cast(const device_ptr<T> &ptr);



} 

#include <hydra/detail/external/hydra_thrust/detail/device_ptr.inl>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>

