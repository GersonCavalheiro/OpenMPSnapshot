



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>

#include <iterator>

#if defined(_MSC_VER) && _MSC_VER < 1916 
#include <vector>
#include <string>
#include <array>

#if HYDRA_THRUST_CPP_DIALECT >= 2017
#include <string_view>
#endif
#endif

HYDRA_THRUST_BEGIN_NS

namespace detail
{

template <typename Iterator>
struct is_contiguous_iterator_impl;

} 

template <typename Iterator>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_contiguous_iterator =
#else
struct is_contiguous_iterator :
#endif
detail::is_contiguous_iterator_impl<Iterator>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename Iterator>
constexpr bool is_contiguous_iterator_v = is_contiguous_iterator<Iterator>::value;
#endif

template <typename Iterator>
struct proclaim_contiguous_iterator : false_type {};

#define HYDRA_THRUST_PROCLAIM_CONTIGUOUS_ITERATOR(Iterator)                         \
HYDRA_THRUST_BEGIN_NS                                                             \
template <>                                                                 \
struct proclaim_contiguous_iterator<Iterator> : ::hydra_thrust::true_type {};     \
HYDRA_THRUST_END_NS                                                               \



namespace detail
{

template <typename Iterator>
struct is_libcxx_wrap_iter : false_type {};

#if defined(_LIBCPP_VERSION)
template <typename Iterator>
struct is_libcxx_wrap_iter<
_VSTD::__wrap_iter<Iterator>
> : true_type {};
#endif

template <typename Iterator>
struct is_libstdcxx_normal_iterator : false_type {};

#if defined(__GLIBCXX__)
template <typename Iterator, typename Container>
struct is_libstdcxx_normal_iterator<
::__gnu_cxx::__normal_iterator<Iterator, Container>
> : true_type {};
#endif

#if   _MSC_VER >= 1916 
template <typename Iterator>
struct is_msvc_contiguous_iterator
: is_pointer<::std::_Unwrapped_t<Iterator> > {};
#elif _MSC_VER >= 1700 
template <typename Iterator>
struct is_msvc_contiguous_iterator : false_type {};

template <typename Vector>
struct is_msvc_contiguous_iterator<
::std::_Vector_const_iterator<Vector>
> : true_type {};

template <typename Vector>
struct is_msvc_contiguous_iterator<
::std::_Vector_iterator<Vector>
> : true_type {};

template <typename String>
struct is_msvc_contiguous_iterator<
::std::_String_const_iterator<String>
> : true_type {};

template <typename String>
struct is_msvc_contiguous_iterator<
::std::_String_iterator<String>
> : true_type {};

template <typename T, std::size_t N>
struct is_msvc_contiguous_iterator<
::std::_Array_const_iterator<T, N>
> : true_type {};

template <typename T, std::size_t N>
struct is_msvc_contiguous_iterator<
::std::_Array_iterator<T, N>
> : true_type {};

#if HYDRA_THRUST_CPP_DIALECT >= 2017
template <typename Traits>
struct is_msvc_contiguous_iterator<
::std::_String_view_iterator<Traits>
> : true_type {};
#endif
#else
template <typename Iterator>
struct is_msvc_contiguous_iterator : false_type {};
#endif


template <typename Iterator>
struct is_contiguous_iterator_impl
: integral_constant<
bool
,    is_pointer<Iterator>::value
|| is_hydra_thrust_pointer<Iterator>::value
|| is_libcxx_wrap_iter<Iterator>::value
|| is_libstdcxx_normal_iterator<Iterator>::value
|| is_msvc_contiguous_iterator<Iterator>::value
|| proclaim_contiguous_iterator<Iterator>::value
>
{};

} 

HYDRA_THRUST_END_NS

