



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/pointer.h>
#include <hydra/detail/external/hydra_thrust/detail/reference.h>

namespace hydra_thrust
{
namespace system
{
namespace omp
{

template<typename> class pointer;

} 
} 
} 




namespace hydra_thrust
{

template<typename Element>
struct iterator_traits<hydra_thrust::system::omp::pointer<Element> >
{
private:
typedef hydra_thrust::system::omp::pointer<Element> ptr;

public:
typedef typename ptr::iterator_category       iterator_category;
typedef typename ptr::value_type              value_type;
typedef typename ptr::difference_type         difference_type;
typedef ptr                                   pointer;
typedef typename ptr::reference               reference;
}; 

} 




namespace hydra_thrust
{
namespace system
{




namespace omp
{

template<typename Element> class reference;



namespace detail
{

template<typename Element>
struct reference_msvc_workaround
{
typedef hydra_thrust::system::omp::reference<Element> type;
}; 

} 





template<typename T>
class pointer
: public hydra_thrust::pointer<
T,
hydra_thrust::system::omp::tag,
hydra_thrust::system::omp::reference<T>,
hydra_thrust::system::omp::pointer<T>
>
{


private:
typedef hydra_thrust::pointer<
T,
hydra_thrust::system::omp::tag,
typename detail::reference_msvc_workaround<T>::type,
hydra_thrust::system::omp::pointer<T>
> super_t;



public:


__host__ __device__
pointer() : super_t() {}

#if HYDRA_THRUST_CPP_DIALECT >= 2011
__host__ __device__
pointer(decltype(nullptr)) : super_t(nullptr) {}
#endif


template<typename OtherT>
__host__ __device__
explicit pointer(OtherT *ptr) : super_t(ptr) {}


template<typename OtherPointer>
__host__ __device__
pointer(const OtherPointer &other,
typename hydra_thrust::detail::enable_if_pointer_is_convertible<
OtherPointer,
pointer
>::type * = 0) : super_t(other) {}


template<typename OtherPointer>
__host__ __device__
explicit
pointer(const OtherPointer &other,
typename hydra_thrust::detail::enable_if_void_pointer_is_system_convertible<
OtherPointer,
pointer
>::type * = 0) : super_t(other) {}


template<typename OtherPointer>
__host__ __device__
typename hydra_thrust::detail::enable_if_pointer_is_convertible<
OtherPointer,
pointer,
pointer &
>::type
operator=(const OtherPointer &other)
{
return super_t::operator=(other);
}

#if HYDRA_THRUST_CPP_DIALECT >= 2011
__host__ __device__
pointer& operator=(decltype(nullptr))
{
super_t::operator=(nullptr);
return *this;
}
#endif
}; 



template<typename T>
class reference
: public hydra_thrust::reference<
T,
hydra_thrust::system::omp::pointer<T>,
hydra_thrust::system::omp::reference<T>
>
{


private:
typedef hydra_thrust::reference<
T,
hydra_thrust::system::omp::pointer<T>,
hydra_thrust::system::omp::reference<T>
> super_t;



public:


typedef typename super_t::value_type value_type;
typedef typename super_t::pointer    pointer;




__host__ __device__
explicit reference(const pointer &ptr)
: super_t(ptr)
{}


template<typename OtherT>
__host__ __device__
reference(const reference<OtherT> &other,
typename hydra_thrust::detail::enable_if_convertible<
typename reference<OtherT>::pointer,
pointer
>::type * = 0)
: super_t(other)
{}


template<typename OtherT>
reference &operator=(const reference<OtherT> &other);


reference &operator=(const value_type &x);
}; 


template<typename T>
__host__ __device__
void swap(reference<T> x, reference<T> y);

} 



} 


namespace omp
{

using hydra_thrust::system::omp::pointer;
using hydra_thrust::system::omp::reference;

} 

} 

#include <hydra/detail/external/hydra_thrust/system/omp/detail/pointer.inl>

