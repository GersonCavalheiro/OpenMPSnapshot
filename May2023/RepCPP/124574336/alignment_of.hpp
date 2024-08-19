

#ifndef BOOST_TT_ALIGNMENT_OF_HPP_INCLUDED
#define BOOST_TT_ALIGNMENT_OF_HPP_INCLUDED

#include <boost/config.hpp>
#include <cstddef>

#include <boost/type_traits/intrinsics.hpp>
#include <boost/type_traits/integral_constant.hpp>

#ifdef BOOST_MSVC
#   pragma warning(push)
#   pragma warning(disable: 4121 4512) 
#endif
#if defined(BOOST_BORLANDC) && (BOOST_BORLANDC < 0x600)
#pragma option push -Vx- -Ve-
#endif

namespace boost {

template <typename T> struct alignment_of;

namespace detail {

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4324) 
#endif
template <typename T>
struct alignment_of_hack
{
char c;
T t;
alignment_of_hack();
};
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <unsigned A, unsigned S>
struct alignment_logic
{
BOOST_STATIC_CONSTANT(std::size_t, value = A < S ? A : S);
};


template< typename T >
struct alignment_of_impl
{
#if defined(BOOST_MSVC) && (BOOST_MSVC >= 1400)
BOOST_STATIC_CONSTANT(std::size_t, value =
(::boost::detail::alignment_logic<
sizeof(::boost::detail::alignment_of_hack<T>) - sizeof(T),
__alignof(T)
>::value));
#elif !defined(BOOST_ALIGNMENT_OF)
BOOST_STATIC_CONSTANT(std::size_t, value =
(::boost::detail::alignment_logic<
sizeof(::boost::detail::alignment_of_hack<T>) - sizeof(T),
sizeof(T)
>::value));
#else
BOOST_STATIC_CONSTANT(std::size_t, value = BOOST_ALIGNMENT_OF(T));
#endif
};

} 

template <class T> struct alignment_of : public integral_constant<std::size_t, ::boost::detail::alignment_of_impl<T>::value>{};

template <typename T> struct alignment_of<T&> : public alignment_of<T*>{};

#ifdef BOOST_BORLANDC
struct long_double_wrapper{ long double ld; };
template<> struct alignment_of<long double> : public alignment_of<long_double_wrapper>{};
#endif

template<> struct alignment_of<void> : integral_constant<std::size_t, 0>{};
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template<> struct alignment_of<void const> : integral_constant<std::size_t, 0>{};
template<> struct alignment_of<void const volatile> : integral_constant<std::size_t, 0>{};
template<> struct alignment_of<void volatile> : integral_constant<std::size_t, 0>{};
#endif

} 

#if defined(BOOST_BORLANDC) && (BOOST_BORLANDC < 0x600)
#pragma option pop
#endif
#ifdef BOOST_MSVC
#   pragma warning(pop)
#endif

#endif 

