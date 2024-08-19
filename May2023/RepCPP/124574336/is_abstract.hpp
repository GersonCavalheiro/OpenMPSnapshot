#ifndef BOOST_TT_IS_ABSTRACT_CLASS_HPP
#define BOOST_TT_IS_ABSTRACT_CLASS_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <cstddef> 
#include <boost/type_traits/intrinsics.hpp>
#include <boost/type_traits/integral_constant.hpp>
#ifndef BOOST_IS_ABSTRACT
#include <boost/static_assert.hpp>
#include <boost/type_traits/detail/yes_no_type.hpp>
#include <boost/type_traits/is_class.hpp>
#ifdef BOOST_NO_IS_ABSTRACT
#include <boost/type_traits/is_polymorphic.hpp>
#endif
#endif

namespace boost {

namespace detail{

#ifdef BOOST_IS_ABSTRACT
template <class T>
struct is_abstract_imp
{
BOOST_STATIC_CONSTANT(bool, value = BOOST_IS_ABSTRACT(T));
};
#elif !defined(BOOST_NO_IS_ABSTRACT)
template<class T>
struct is_abstract_imp2
{
template<class U>
static type_traits::no_type check_sig(U (*)[1]);
template<class U>
static type_traits::yes_type check_sig(...);
BOOST_STATIC_ASSERT(sizeof(T) != 0);

#ifdef __GNUC__
BOOST_STATIC_CONSTANT(std::size_t, s1 = sizeof(is_abstract_imp2<T>::template check_sig<T>(0)));
#else
#if BOOST_WORKAROUND(BOOST_MSVC_FULL_VER, >= 140050000)
#pragma warning(push)
#pragma warning(disable:6334)
#endif
BOOST_STATIC_CONSTANT(std::size_t, s1 = sizeof(check_sig<T>(0)));
#if BOOST_WORKAROUND(BOOST_MSVC_FULL_VER, >= 140050000)
#pragma warning(pop)
#endif
#endif

BOOST_STATIC_CONSTANT(bool, value = 
(s1 == sizeof(type_traits::yes_type)));
};

template <bool v>
struct is_abstract_select
{
template <class T>
struct rebind
{
typedef is_abstract_imp2<T> type;
};
};
template <>
struct is_abstract_select<false>
{
template <class T>
struct rebind
{
typedef false_type type;
};
};

template <class T>
struct is_abstract_imp
{
typedef is_abstract_select< ::boost::is_class<T>::value> selector;
typedef typename selector::template rebind<T> binder;
typedef typename binder::type type;

BOOST_STATIC_CONSTANT(bool, value = type::value);
};

#endif
}

#ifndef BOOST_NO_IS_ABSTRACT
template <class T> struct is_abstract : public integral_constant<bool, ::boost::detail::is_abstract_imp<T>::value> {};
#else
template <class T> struct is_abstract : public integral_constant<bool, ::boost::detail::is_polymorphic_imp<T>::value> {};
#endif

} 

#endif 
