#ifndef BOOST_BIND_DETAIL_RESULT_TRAITS_HPP_INCLUDED
#define BOOST_BIND_DETAIL_RESULT_TRAITS_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#if defined(_MSVC_LANG) && _MSVC_LANG >= 17
#include <functional>
#endif

namespace boost
{

namespace _bi
{

template<class R, class F> struct result_traits
{
typedef R type;
};

struct unspecified {};

template<class F> struct result_traits<unspecified, F>
{
typedef typename F::result_type type;
};

template<class F> struct result_traits< unspecified, reference_wrapper<F> >
{
typedef typename F::result_type type;
};

#if defined(_MSVC_LANG) && _MSVC_LANG >= 17

template<class T> struct result_traits< unspecified, std::plus<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::minus<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::multiplies<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::divides<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::modulus<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::negate<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::equal_to<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::not_equal_to<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::greater<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::less<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::greater_equal<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::less_equal<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::logical_and<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::logical_or<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::logical_not<T> >
{
typedef bool type;
};

template<class T> struct result_traits< unspecified, std::bit_and<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::bit_or<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::bit_xor<T> >
{
typedef T type;
};

template<class T> struct result_traits< unspecified, std::bit_not<T> >
{
typedef T type;
};

#endif

} 

} 

#endif 
