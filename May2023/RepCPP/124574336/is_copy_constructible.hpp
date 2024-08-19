
#ifndef BOOST_TT_IS_COPY_CONSTRUCTIBLE_HPP_INCLUDED
#define BOOST_TT_IS_COPY_CONSTRUCTIBLE_HPP_INCLUDED

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_DECLTYPE) && !BOOST_WORKAROUND(BOOST_MSVC, < 1800) && !BOOST_WORKAROUND(BOOST_GCC_VERSION, < 40900)

#include <boost/type_traits/is_constructible.hpp>

#if !BOOST_WORKAROUND(BOOST_MSVC, <= 1800)

namespace boost {

template <class T> struct is_copy_constructible : public boost::is_constructible<T, const T&>{};

template <> struct is_copy_constructible<void> : public false_type{};
template <> struct is_copy_constructible<void const> : public false_type{};
template <> struct is_copy_constructible<void const volatile> : public false_type{};
template <> struct is_copy_constructible<void volatile> : public false_type{};

} 

#else
#include <boost/type_traits/is_noncopyable.hpp>

namespace boost {

namespace detail
{

template <class T, bool b> struct is_copy_constructible_imp : public boost::is_constructible<T, const T&>{};
template <class T> struct is_copy_constructible_imp<T, true> : public false_type{};

}

template <class T> struct is_copy_constructible : public detail::is_copy_constructible_imp<T, is_noncopyable<T>::value>{};

template <> struct is_copy_constructible<void> : public false_type{};
template <> struct is_copy_constructible<void const> : public false_type{};
template <> struct is_copy_constructible<void const volatile> : public false_type{};
template <> struct is_copy_constructible<void volatile> : public false_type{};

} 

#endif

#else

#include <boost/type_traits/detail/yes_no_type.hpp>
#include <boost/type_traits/is_noncopyable.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/is_rvalue_reference.hpp>
#include <boost/type_traits/declval.hpp>
#include <boost/type_traits/is_array.hpp>
#include <boost/type_traits/declval.hpp>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4181)
#endif

namespace boost {

namespace detail{

template <bool DerivedFromNoncopyable, class T>
struct is_copy_constructible_impl2 {

#if !defined(BOOST_NO_CXX11_DELETED_FUNCTIONS) && !defined(BOOST_INTEL_CXX_VERSION) && !(defined(BOOST_MSVC) && _MSC_VER == 1800)

#ifdef BOOST_NO_CXX11_DECLTYPE
template <class T1>
static boost::type_traits::yes_type test(const T1&, boost::mpl::int_<sizeof(T1(boost::declval<const T1&>()))>* = 0);
#else
template <class T1>
static boost::type_traits::yes_type test(const T1&, decltype(T1(boost::declval<const T1&>()))* = 0);
#endif

static boost::type_traits::no_type test(...);
#else
template <class T1>
static boost::type_traits::no_type test(const T1&, typename T1::boost_move_no_copy_constructor_or_assign* = 0);
static boost::type_traits::yes_type test(...);
#endif

BOOST_STATIC_CONSTANT(bool, value = (
sizeof(test(
boost::declval<BOOST_DEDUCED_TYPENAME boost::add_reference<T const>::type>()
)) == sizeof(boost::type_traits::yes_type)
&&
!boost::is_rvalue_reference<T>::value
&& !boost::is_array<T>::value
));
};

template <class T>
struct is_copy_constructible_impl2<true, T> {
BOOST_STATIC_CONSTANT(bool, value = false);
};

template <class T>
struct is_copy_constructible_impl {

BOOST_STATIC_CONSTANT(bool, value = (
boost::detail::is_copy_constructible_impl2<
boost::is_noncopyable<T>::value,
T
>::value
));
};

} 

template <class T> struct is_copy_constructible : public integral_constant<bool, ::boost::detail::is_copy_constructible_impl<T>::value>{};
template <> struct is_copy_constructible<void> : public false_type{};
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <> struct is_copy_constructible<void const> : public false_type{};
template <> struct is_copy_constructible<void volatile> : public false_type{};
template <> struct is_copy_constructible<void const volatile> : public false_type{};
#endif

} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif

#endif 
