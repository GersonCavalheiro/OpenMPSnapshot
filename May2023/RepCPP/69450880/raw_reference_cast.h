

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/has_nested_type.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/tuple/tuple_transform.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/tuple_of_iterator_references.h>



namespace hydra_thrust
{
namespace detail
{


__HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(is_wrapped_reference, wrapped_reference_hint)


template<typename T>
struct is_unwrappable
: is_wrapped_reference<T>
{};

#ifdef HYDRA_THRUST_VARIADIC_TUPLE
template<typename... Types>
struct is_unwrappable<
hydra_thrust::tuple<Types...>
>
: or_<
is_unwrappable<Types>...
>
{};


template<
typename... Types
>
struct is_unwrappable<
hydra_thrust::detail::tuple_of_iterator_references<Types...>
>
: or_<
is_unwrappable<Types>...
>
{};
#else

template<
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
struct is_unwrappable<
hydra_thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>
: or_<
is_unwrappable<T0>,
is_unwrappable<T1>,
is_unwrappable<T2>,
is_unwrappable<T3>,
is_unwrappable<T4>,
is_unwrappable<T5>,
is_unwrappable<T6>,
is_unwrappable<T7>,
is_unwrappable<T8>,
is_unwrappable<T9>
>
{};

template<
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
struct is_unwrappable<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>
: or_<
is_unwrappable<T0>,
is_unwrappable<T1>,
is_unwrappable<T2>,
is_unwrappable<T3>,
is_unwrappable<T4>,
is_unwrappable<T5>,
is_unwrappable<T6>,
is_unwrappable<T7>,
is_unwrappable<T8>,
is_unwrappable<T9>
>
{};

#endif


template<typename T, typename Result = void>
struct enable_if_unwrappable
: enable_if<
is_unwrappable<T>::value,
Result
>
{};


namespace raw_reference_detail
{


template<typename T, typename Enable = void>
struct raw_reference_impl
: add_reference<T>
{};


template<typename T>
struct raw_reference_impl<
T,
typename hydra_thrust::detail::enable_if<
is_wrapped_reference<
typename remove_cv<T>::type
>::value
>::type
>
{
typedef typename add_reference<
typename pointer_element<typename T::pointer>::type
>::type type;
};


} 


template<typename T>
struct raw_reference :
raw_reference_detail::raw_reference_impl<T>
{};


namespace raw_reference_detail
{


template<typename T>
struct raw_reference_tuple_helper
: eval_if<
is_unwrappable<
typename remove_cv<T>::type
>::value,
raw_reference<T>,
identity_<T>
>
{};

#ifdef HYDRA_THRUST_VARIADIC_TUPLE

template <
typename... Types
>
struct raw_reference_tuple_helper<
hydra_thrust::tuple<Types...>
>
{
typedef hydra_thrust::tuple<
typename raw_reference_tuple_helper<Types>::type...
> type;
};


template <
typename... Types
>
struct raw_reference_tuple_helper<
hydra_thrust::detail::tuple_of_iterator_references<Types...>
>
{
typedef hydra_thrust::detail::tuple_of_iterator_references<
typename raw_reference_tuple_helper<Types>::type...
> type;
};

#else
template <
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
struct raw_reference_tuple_helper<
hydra_thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>
{
typedef hydra_thrust::tuple<
typename raw_reference_tuple_helper<T0>::type,
typename raw_reference_tuple_helper<T1>::type,
typename raw_reference_tuple_helper<T2>::type,
typename raw_reference_tuple_helper<T3>::type,
typename raw_reference_tuple_helper<T4>::type,
typename raw_reference_tuple_helper<T5>::type,
typename raw_reference_tuple_helper<T6>::type,
typename raw_reference_tuple_helper<T7>::type,
typename raw_reference_tuple_helper<T8>::type,
typename raw_reference_tuple_helper<T9>::type
> type;
};


template <
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
struct raw_reference_tuple_helper<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>
{
typedef hydra_thrust::detail::tuple_of_iterator_references<
typename raw_reference_tuple_helper<T0>::type,
typename raw_reference_tuple_helper<T1>::type,
typename raw_reference_tuple_helper<T2>::type,
typename raw_reference_tuple_helper<T3>::type,
typename raw_reference_tuple_helper<T4>::type,
typename raw_reference_tuple_helper<T5>::type,
typename raw_reference_tuple_helper<T6>::type,
typename raw_reference_tuple_helper<T7>::type,
typename raw_reference_tuple_helper<T8>::type,
typename raw_reference_tuple_helper<T9>::type
> type;
};
#endif
} 

#ifdef HYDRA_THRUST_VARIADIC_TUPLE


template <
typename... Types
>
struct raw_reference<
hydra_thrust::tuple<Types...>
>
{
private:
typedef hydra_thrust::tuple<Types...> tuple_type;

public:
typedef typename eval_if<
is_unwrappable<tuple_type>::value,
raw_reference_detail::raw_reference_tuple_helper<tuple_type>,
add_reference<tuple_type>
>::type type;
};


template <
typename... Types
>
struct raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<Types...>
>
{
private:
typedef detail::tuple_of_iterator_references<Types...> tuple_type;

public:
typedef typename raw_reference_detail::raw_reference_tuple_helper<tuple_type>::type type;

};
#else


template <
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
struct raw_reference<
hydra_thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>
{
private:
typedef hydra_thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> tuple_type;

public:
typedef typename eval_if<
is_unwrappable<tuple_type>::value,
raw_reference_detail::raw_reference_tuple_helper<tuple_type>,
add_reference<tuple_type>
>::type type;
};


template <
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
struct raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>
{
private:
typedef detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> tuple_type;

public:
typedef typename raw_reference_detail::raw_reference_tuple_helper<tuple_type>::type type;

};
#endif

} 


template<typename T>
inline __host__ __device__
typename detail::raw_reference<T>::type
raw_reference_cast(T &ref);


template<typename T>
inline __host__ __device__
typename detail::raw_reference<const T>::type
raw_reference_cast(const T &ref);

#ifdef HYDRA_THRUST_VARIADIC_TUPLE

template<
typename... Types
>
__host__ __device__
typename detail::enable_if_unwrappable<
hydra_thrust::detail::tuple_of_iterator_references<Types...>,
typename detail::raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<Types...>
>::type
>::type
raw_reference_cast(hydra_thrust::detail::tuple_of_iterator_references<Types...> t);

#else

template<
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
__host__ __device__
typename detail::enable_if_unwrappable<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>,
typename detail::raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>::type
>::type
raw_reference_cast(hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> t);

#endif

namespace detail
{


struct raw_reference_caster
{
template<typename T>
__host__ __device__
typename detail::raw_reference<T>::type operator()(T &ref)
{
return hydra_thrust::raw_reference_cast(ref);
}

template<typename T>
__host__ __device__
typename detail::raw_reference<const T>::type operator()(const T &ref)
{
return hydra_thrust::raw_reference_cast(ref);
}

#ifdef HYDRA_THRUST_VARIADIC_TUPLE
template<
typename... Types
>
__host__ __device__
typename detail::raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<Types...>
>::type
operator()(hydra_thrust::detail::tuple_of_iterator_references<Types...> t,
typename enable_if<
is_unwrappable<hydra_thrust::detail::tuple_of_iterator_references<Types...> >::value
>::type * = 0)
{
return hydra_thrust::raw_reference_cast(t);
}

#else

template<
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
__host__ __device__
typename detail::raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>::type
operator()(hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> t,
typename enable_if<
is_unwrappable<hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> >::value
>::type * = 0)
{
return hydra_thrust::raw_reference_cast(t);
}
#endif

}; 


} 


template<typename T>
inline __host__ __device__
typename detail::raw_reference<T>::type
raw_reference_cast(T &ref)
{
return *hydra_thrust::raw_pointer_cast(&ref);
} 


template<typename T>
inline __host__ __device__
typename detail::raw_reference<const T>::type
raw_reference_cast(const T &ref)
{
return *hydra_thrust::raw_pointer_cast(&ref);
} 

#ifdef HYDRA_THRUST_VARIADIC_TUPLE

template<
typename... Types
>
__host__ __device__
typename detail::enable_if_unwrappable<
hydra_thrust::detail::tuple_of_iterator_references<Types...>,
typename detail::raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<Types...>
>::type
>::type
raw_reference_cast(hydra_thrust::detail::tuple_of_iterator_references<Types...> t)
{
hydra_thrust::detail::raw_reference_caster f;

return hydra_thrust::detail::tuple_host_device_transform<detail::raw_reference_detail::raw_reference_tuple_helper>(t, f);
} 

#else

template<
typename T0, typename T1, typename T2,
typename T3, typename T4, typename T5,
typename T6, typename T7, typename T8,
typename T9
>
__host__ __device__
typename detail::enable_if_unwrappable<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>,
typename detail::raw_reference<
hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
>::type
>::type
raw_reference_cast(hydra_thrust::detail::tuple_of_iterator_references<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> t)
{
hydra_thrust::detail::raw_reference_caster f;

return hydra_thrust::detail::tuple_host_device_transform<detail::raw_reference_detail::raw_reference_tuple_helper>(t, f);
} 

#endif
} 

