



#ifndef BOOST_MATH_PROMOTION_HPP
#define BOOST_MATH_PROMOTION_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/type_traits/is_floating_point.hpp> 
#include <boost/type_traits/is_integral.hpp> 
#include <boost/type_traits/is_convertible.hpp> 
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/mpl/if.hpp> 
#include <boost/mpl/and.hpp> 
#include <boost/mpl/or.hpp> 
#include <boost/mpl/not.hpp> 

#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
#include <boost/static_assert.hpp>
#endif

namespace boost
{
namespace math
{
namespace tools
{


template <class T>
struct promote_arg
{ 
typedef typename mpl::if_<is_integral<T>, double, T>::type type;
};
template <> struct promote_arg<float> { typedef float type; };
template <> struct promote_arg<double>{ typedef double type; };
template <> struct promote_arg<long double> { typedef long double type; };
template <> struct promote_arg<int> {  typedef double type; };

template <class T1, class T2>
struct promote_args_2
{ 
typedef typename promote_arg<T1>::type T1P; 
typedef typename promote_arg<T2>::type T2P; 

typedef typename mpl::if_c<
is_floating_point<T1P>::value && is_floating_point<T2P>::value, 
#ifdef BOOST_MATH_USE_FLOAT128
typename mpl::if_c<is_same<__float128, T1P>::value || is_same<__float128, T2P>::value, 
__float128,
#endif
typename mpl::if_c<is_same<long double, T1P>::value || is_same<long double, T2P>::value, 
long double, 
typename mpl::if_c<is_same<double, T1P>::value || is_same<double, T2P>::value, 
double, 
float 
>::type
#ifdef BOOST_MATH_USE_FLOAT128
>::type
#endif
>::type,
typename mpl::if_c<!is_floating_point<T2P>::value && ::boost::is_convertible<T1P, T2P>::value, T2P, T1P>::type>::type type;
}; 
template <> struct promote_args_2<float, float> { typedef float type; };
template <> struct promote_args_2<double, double>{ typedef double type; };
template <> struct promote_args_2<long double, long double> { typedef long double type; };
template <> struct promote_args_2<int, int> {  typedef double type; };
template <> struct promote_args_2<int, float> {  typedef double type; };
template <> struct promote_args_2<float, int> {  typedef double type; };
template <> struct promote_args_2<int, double> {  typedef double type; };
template <> struct promote_args_2<double, int> {  typedef double type; };
template <> struct promote_args_2<int, long double> {  typedef long double type; };
template <> struct promote_args_2<long double, int> {  typedef long double type; };
template <> struct promote_args_2<float, double> {  typedef double type; };
template <> struct promote_args_2<double, float> {  typedef double type; };
template <> struct promote_args_2<float, long double> {  typedef long double type; };
template <> struct promote_args_2<long double, float> {  typedef long double type; };
template <> struct promote_args_2<double, long double> {  typedef long double type; };
template <> struct promote_args_2<long double, double> {  typedef long double type; };

template <class T1, class T2=float, class T3=float, class T4=float, class T5=float, class T6=float>
struct promote_args
{
typedef typename promote_args_2<
typename remove_cv<T1>::type,
typename promote_args_2<
typename remove_cv<T2>::type,
typename promote_args_2<
typename remove_cv<T3>::type,
typename promote_args_2<
typename remove_cv<T4>::type,
typename promote_args_2<
typename remove_cv<T5>::type, typename remove_cv<T6>::type
>::type
>::type
>::type
>::type
>::type type;

#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
BOOST_STATIC_ASSERT_MSG((0 == ::boost::is_same<type, long double>::value), "Sorry, but this platform does not have sufficient long double support for the special functions to be reliably implemented.");
#endif
};

template <class T1, class T2=float, class T3=float, class T4=float, class T5=float, class T6=float>
struct promote_args_permissive
{
typedef typename promote_args_2<
typename remove_cv<T1>::type,
typename promote_args_2<
typename remove_cv<T2>::type,
typename promote_args_2<
typename remove_cv<T3>::type,
typename promote_args_2<
typename remove_cv<T4>::type,
typename promote_args_2<
typename remove_cv<T5>::type, typename remove_cv<T6>::type
>::type
>::type
>::type
>::type
>::type type;
};

} 
} 
} 

#endif 

