


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/static_assert.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/type_traits/is_contiguous_iterator.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#include <type_traits>
#endif

HYDRA_THRUST_BEGIN_NS

namespace detail
{

template <typename T>
struct is_trivially_relocatable_impl;

} 

template <typename T>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_trivially_relocatable =
#else
struct is_trivially_relocatable :
#endif
detail::is_trivially_relocatable_impl<T>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename T>
constexpr bool is_trivially_relocatable_v = is_trivially_relocatable<T>::value;
#endif

template <typename From, typename To>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_trivially_relocatable_to =
#else
struct is_trivially_relocatable_to :
#endif
integral_constant<
bool
, detail::is_same<From, To>::value && is_trivially_relocatable<To>::value
>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename From, typename To>
constexpr bool is_trivially_relocatable_to_v
= is_trivially_relocatable_to<From, To>::value;
#endif

template <typename FromIterator, typename ToIterator>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_indirectly_trivially_relocatable_to =
#else
struct is_indirectly_trivially_relocatable_to :
#endif
integral_constant<
bool
,    is_contiguous_iterator<FromIterator>::value
&& is_contiguous_iterator<ToIterator>::value
&& is_trivially_relocatable_to<
typename hydra_thrust::iterator_traits<FromIterator>::value_type,
typename hydra_thrust::iterator_traits<ToIterator>::value_type
>::value
>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename FromIterator, typename ToIterator>
constexpr bool is_trivial_relocatable_sequence_copy_v
= is_indirectly_trivially_relocatable_to<FromIterator, ToIterator>::value;
#endif

template <typename T>
struct proclaim_trivially_relocatable : false_type {};

#define HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(T)                              \
HYDRA_THRUST_BEGIN_NS                                                             \
template <>                                                                 \
struct proclaim_trivially_relocatable<T> : ::hydra_thrust::true_type {};          \
HYDRA_THRUST_END_NS                                                               \



namespace detail
{


#ifndef __has_feature
#define __has_feature(x) 0
#endif

template <typename T>
struct is_trivially_copyable_impl
: integral_constant<
bool,
#if HYDRA_THRUST_CPP_DIALECT >= 2011
#if defined(__GLIBCXX__) && __has_feature(is_trivially_copyable)
__is_trivially_copyable(T)
#elif HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC && HYDRA_THRUST_GCC_VERSION >= 50000
std::is_trivially_copyable<T>::value
#else
has_trivial_assign<T>::value
#endif
#else
has_trivial_assign<T>::value
#endif
>
{
};

template <typename T>
struct is_trivially_relocatable_impl
: integral_constant<
bool,
is_trivially_copyable_impl<T>::value
|| proclaim_trivially_relocatable<T>::value
>
{};

template <typename T, std::size_t N>
struct is_trivially_relocatable_impl<T[N]> : is_trivially_relocatable_impl<T> {};

} 

HYDRA_THRUST_END_NS

#if HYDRA_THRUST_DEVICE_SYSTEM == HYDRA_THRUST_DEVICE_SYSTEM_CUDA

#include <hydra/detail/external/hydra_thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(char4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uchar4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(short4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ushort4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(int4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(uint4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(long4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulong4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(longlong4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(ulonglong4)

struct __half;
struct __half2;

HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(__half)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(__half2)

HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(float4)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double1)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double2)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double3)
HYDRA_THRUST_PROCLAIM_TRIVIALLY_RELOCATABLE(double4)
#endif

