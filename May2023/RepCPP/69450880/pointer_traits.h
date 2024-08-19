

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/is_metafunction_defined.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/has_nested_type.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <cstddef>

namespace hydra_thrust
{
namespace detail
{

template<typename Ptr> struct pointer_element;

template<template<typename> class Ptr, typename Arg>
struct pointer_element<Ptr<Arg> >
{
typedef Arg type;
};

template<template<typename,typename> class Ptr, typename Arg1, typename Arg2>
struct pointer_element<Ptr<Arg1,Arg2> >
{
typedef Arg1 type;
};

template<template<typename,typename,typename> class Ptr, typename Arg1, typename Arg2, typename Arg3>
struct pointer_element<Ptr<Arg1,Arg2,Arg3> >
{
typedef Arg1 type;
};

template<template<typename,typename,typename,typename> class Ptr, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
struct pointer_element<Ptr<Arg1,Arg2,Arg3,Arg4> >
{
typedef Arg1 type;
};

template<template<typename,typename,typename,typename,typename> class Ptr, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
struct pointer_element<Ptr<Arg1,Arg2,Arg3,Arg4,Arg5> >
{
typedef Arg1 type;
};

template<typename T>
struct pointer_element<T*>
{
typedef T type;
};

template<typename Ptr>
struct pointer_difference
{
typedef typename Ptr::difference_type type;
};

template<typename T>
struct pointer_difference<T*>
{
typedef std::ptrdiff_t type;
};

template<typename Ptr, typename T> struct rebind_pointer;

template<typename T, typename U>
struct rebind_pointer<T*,U>
{
typedef U* type;
};

template<template<typename> class Ptr, typename Arg, typename T>
struct rebind_pointer<Ptr<Arg>,T>
{
typedef Ptr<T> type;
};

template<template<typename, typename> class Ptr, typename Arg1, typename Arg2, typename T>
struct rebind_pointer<Ptr<Arg1,Arg2>,T>
{
typedef Ptr<T,Arg2> type;
};

template<template<typename, typename, typename> class Ptr, typename Arg1, typename Arg2, typename Arg3, typename T>
struct rebind_pointer<Ptr<Arg1,Arg2,Arg3>,T>
{
typedef Ptr<T,Arg2,Arg3> type;
};

template<template<typename, typename, typename, typename> class Ptr, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename T>
struct rebind_pointer<Ptr<Arg1,Arg2,Arg3,Arg4>,T>
{
typedef Ptr<T,Arg2,Arg3,Arg4> type;
};

__HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(has_raw_pointer, raw_pointer)

namespace pointer_traits_detail
{

template<typename Ptr, typename Enable = void> struct pointer_raw_pointer_impl {};

template<typename T>
struct pointer_raw_pointer_impl<T*>
{
typedef T* type;
};

template<typename Ptr>
struct pointer_raw_pointer_impl<Ptr, typename enable_if<has_raw_pointer<Ptr>::value>::type>
{
typedef typename Ptr::raw_pointer type;
};

} 

template<typename T>
struct pointer_raw_pointer
: pointer_traits_detail::pointer_raw_pointer_impl<T>
{};

namespace pointer_traits_detail
{

template<typename Void>
struct capture_address
{
template<typename T>
__host__ __device__
capture_address(T &r)
: m_addr(&r)
{}

inline __host__ __device__
Void *operator&() const
{
return m_addr;
}

Void *m_addr;
};

template<typename T>
struct pointer_to_param
: hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_void<T>::value,
hydra_thrust::detail::identity_<capture_address<T> >,
hydra_thrust::detail::add_reference<T>
>
{};

}

template<typename Ptr>
struct pointer_traits
{
typedef Ptr                                    pointer;
typedef typename Ptr::reference                reference;
typedef typename pointer_element<Ptr>::type    element_type;
typedef typename pointer_difference<Ptr>::type difference_type;

template<typename U>
struct rebind 
{
typedef typename rebind_pointer<Ptr,U>::type other;
};

__host__ __device__
inline static pointer pointer_to(typename pointer_traits_detail::pointer_to_param<element_type>::type r)
{

return pointer(&r);
}

typedef typename pointer_raw_pointer<Ptr>::type raw_pointer;

__host__ __device__
inline static raw_pointer get(pointer ptr)
{
return ptr.get();
}
};

template<typename T>
struct pointer_traits<T*>
{
typedef T*                                    pointer;
typedef T&                                    reference;
typedef T                                     element_type;
typedef typename pointer_difference<T*>::type difference_type;

template<typename U>
struct rebind
{
typedef U* other;
};

__host__ __device__
inline static pointer pointer_to(typename pointer_traits_detail::pointer_to_param<element_type>::type r)
{
return &r;
}

typedef typename pointer_raw_pointer<T*>::type raw_pointer;

__host__ __device__
inline static raw_pointer get(pointer ptr)
{
return ptr;
}
};

template<>
struct pointer_traits<void*>
{
typedef void*                                    pointer;
typedef void                                     reference;
typedef void                                     element_type;
typedef pointer_difference<void*>::type          difference_type;

template<typename U>
struct rebind
{
typedef U* other;
};

__host__ __device__
inline static pointer pointer_to(pointer_traits_detail::pointer_to_param<element_type>::type r)
{
return &r;
}

typedef pointer_raw_pointer<void*>::type raw_pointer;

__host__ __device__
inline static raw_pointer get(pointer ptr)
{
return ptr;
}
};

template<>
struct pointer_traits<const void*>
{
typedef const void*                           pointer;
typedef const void                            reference;
typedef const void                            element_type;
typedef pointer_difference<const void*>::type difference_type;

template<typename U>
struct rebind
{
typedef U* other;
};

__host__ __device__
inline static pointer pointer_to(pointer_traits_detail::pointer_to_param<element_type>::type r)
{
return &r;
}

typedef pointer_raw_pointer<const void*>::type raw_pointer;

__host__ __device__
inline static raw_pointer get(pointer ptr)
{
return ptr;
}
};

template<typename FromPtr, typename ToPtr>
struct is_pointer_system_convertible
: hydra_thrust::detail::is_convertible<
typename iterator_system<FromPtr>::type,
typename iterator_system<ToPtr>::type
>
{};

template<typename FromPtr, typename ToPtr>
struct is_pointer_convertible
: hydra_thrust::detail::and_<
hydra_thrust::detail::is_convertible<
typename pointer_element<FromPtr>::type *,
typename pointer_element<ToPtr>::type *
>,
is_pointer_system_convertible<FromPtr, ToPtr>
>
{};

template<typename FromPtr, typename ToPtr>
struct is_void_pointer_system_convertible
: hydra_thrust::detail::and_<
hydra_thrust::detail::is_same<
typename pointer_element<FromPtr>::type,
void
>,
is_pointer_system_convertible<FromPtr, ToPtr>
>
{};

template<typename T>
struct is_hydra_thrust_pointer
: is_metafunction_defined<pointer_raw_pointer<T> >
{};

template<typename FromPtr, typename ToPtr>
struct lazy_is_pointer_convertible
: hydra_thrust::detail::eval_if<
is_hydra_thrust_pointer<FromPtr>::value && is_hydra_thrust_pointer<ToPtr>::value,
is_pointer_convertible<FromPtr,ToPtr>,
hydra_thrust::detail::identity_<hydra_thrust::detail::false_type>
>
{};

template<typename FromPtr, typename ToPtr>
struct lazy_is_void_pointer_system_convertible
: hydra_thrust::detail::eval_if<
is_hydra_thrust_pointer<FromPtr>::value && is_hydra_thrust_pointer<ToPtr>::value,
is_void_pointer_system_convertible<FromPtr,ToPtr>,
hydra_thrust::detail::identity_<hydra_thrust::detail::false_type>
>
{};

template<typename FromPtr, typename ToPtr, typename T = void>
struct enable_if_pointer_is_convertible
: hydra_thrust::detail::enable_if<
lazy_is_pointer_convertible<FromPtr,ToPtr>::type::value,
T
>
{};

template<typename FromPtr, typename ToPtr, typename T = void>
struct enable_if_void_pointer_is_system_convertible
: hydra_thrust::detail::enable_if<
lazy_is_void_pointer_system_convertible<FromPtr,ToPtr>::type::value,
T
>
{};


} 
} 

