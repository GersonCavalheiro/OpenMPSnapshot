#ifndef BOOST_MP11_FUNCTION_HPP_INCLUDED
#define BOOST_MP11_FUNCTION_HPP_INCLUDED


#include <boost/mp11/integral.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/mp11/detail/mp_list.hpp>
#include <boost/mp11/detail/mp_count.hpp>
#include <boost/mp11/detail/mp_plus.hpp>
#include <boost/mp11/detail/mp_min_element.hpp>
#include <boost/mp11/detail/mp_void.hpp>
#include <boost/mp11/detail/config.hpp>
#include <type_traits>

namespace boost
{
namespace mp11
{


#if BOOST_MP11_WORKAROUND( BOOST_MP11_MSVC, < 1910 )

namespace detail
{

template<class... T> struct mp_and_impl;

} 

template<class... T> using mp_and = mp_to_bool< typename detail::mp_and_impl<T...>::type >;

namespace detail
{

template<> struct mp_and_impl<>
{
using type = mp_true;
};

template<class T> struct mp_and_impl<T>
{
using type = T;
};

template<class T1, class... T> struct mp_and_impl<T1, T...>
{
using type = mp_eval_if< mp_not<T1>, T1, mp_and, T... >;
};

} 

#else

namespace detail
{

template<class L, class E = void> struct mp_and_impl
{
using type = mp_false;
};

template<class... T> struct mp_and_impl< mp_list<T...>, mp_void<mp_if<T, void>...> >
{
using type = mp_true;
};

} 

template<class... T> using mp_and = typename detail::mp_and_impl<mp_list<T...>>::type;

#endif

#if BOOST_MP11_WORKAROUND( BOOST_MP11_MSVC, < 1920 ) || BOOST_MP11_WORKAROUND( BOOST_MP11_GCC, != 0 )

template<class... T> using mp_all = mp_bool< mp_count_if< mp_list<T...>, mp_not >::value == 0 >;

#elif defined( BOOST_MP11_HAS_FOLD_EXPRESSIONS )

template<class... T> using mp_all = mp_bool<(static_cast<bool>(T::value) && ...)>;

#else

template<class... T> using mp_all = mp_and<mp_to_bool<T>...>;

#endif

namespace detail
{

template<class... T> struct mp_or_impl;

} 

template<class... T> using mp_or = mp_to_bool< typename detail::mp_or_impl<T...>::type >;

namespace detail
{

template<> struct mp_or_impl<>
{
using type = mp_false;
};

template<class T> struct mp_or_impl<T>
{
using type = T;
};

template<class T1, class... T> struct mp_or_impl<T1, T...>
{
using type = mp_eval_if< T1, T1, mp_or, T... >;
};

} 

#if defined( BOOST_MP11_HAS_FOLD_EXPRESSIONS ) && !BOOST_MP11_WORKAROUND( BOOST_MP11_GCC, != 0 ) && !BOOST_MP11_WORKAROUND( BOOST_MP11_MSVC, < 1920 )

template<class... T> using mp_any = mp_bool<(static_cast<bool>(T::value) || ...)>;

#else

template<class... T> using mp_any = mp_bool< mp_count_if< mp_list<T...>, mp_to_bool >::value != 0 >;

#endif

namespace detail
{

template<class... T> struct mp_same_impl;

template<> struct mp_same_impl<>
{
using type = mp_true;
};

template<class T1, class... T> struct mp_same_impl<T1, T...>
{
using type = mp_all<std::is_same<T1, T>...>;
};

} 

template<class... T> using mp_same = typename detail::mp_same_impl<T...>::type;

namespace detail
{

template<class... T> struct mp_similar_impl;

template<> struct mp_similar_impl<>
{
using type = mp_true;
};

template<class T> struct mp_similar_impl<T>
{
using type = mp_true;
};

template<class T> struct mp_similar_impl<T, T>
{
using type = mp_true;
};

template<class T1, class T2> struct mp_similar_impl<T1, T2>
{
using type = mp_false;
};

template<template<class...> class L, class... T1, class... T2> struct mp_similar_impl<L<T1...>, L<T2...>>
{
using type = mp_true;
};

template<template<class...> class L, class... T> struct mp_similar_impl<L<T...>, L<T...>>
{
using type = mp_true;
};

template<class T1, class T2, class T3, class... T> struct mp_similar_impl<T1, T2, T3, T...>
{
using type = mp_all< typename mp_similar_impl<T1, T2>::type, typename mp_similar_impl<T1, T3>::type, typename mp_similar_impl<T1, T>::type... >;
};

} 

template<class... T> using mp_similar = typename detail::mp_similar_impl<T...>::type;

#if BOOST_MP11_GCC
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wsign-compare"
#endif

template<class T1, class T2> using mp_less = mp_bool<(T1::value < 0 && T2::value >= 0) || ((T1::value < T2::value) && !(T1::value >= 0 && T2::value < 0))>;

#if BOOST_MP11_GCC
# pragma GCC diagnostic pop
#endif

template<class T1, class... T> using mp_min = mp_min_element<mp_list<T1, T...>, mp_less>;

template<class T1, class... T> using mp_max = mp_max_element<mp_list<T1, T...>, mp_less>;

} 
} 

#endif 
