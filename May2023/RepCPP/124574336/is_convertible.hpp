

#ifndef BOOST_TT_IS_CONVERTIBLE_HPP_INCLUDED
#define BOOST_TT_IS_CONVERTIBLE_HPP_INCLUDED

#include <boost/type_traits/intrinsics.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_complete.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/type_traits/is_array.hpp>
#include <boost/static_assert.hpp>
#ifndef BOOST_IS_CONVERTIBLE
#include <boost/type_traits/detail/yes_no_type.hpp>
#include <boost/type_traits/detail/config.hpp>
#include <boost/type_traits/is_array.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/is_void.hpp>
#if !defined(BOOST_NO_IS_ABSTRACT)
#include <boost/type_traits/is_abstract.hpp>
#endif
#include <boost/type_traits/add_lvalue_reference.hpp>
#include <boost/type_traits/add_rvalue_reference.hpp>
#include <boost/type_traits/is_function.hpp>

#if defined(__MWERKS__)
#include <boost/type_traits/remove_reference.hpp>
#endif
#if !defined(BOOST_NO_SFINAE_EXPR) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#  include <boost/type_traits/declval.hpp>
#endif
#elif defined(BOOST_MSVC) || defined(BOOST_INTEL)
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/is_same.hpp>
#endif 

namespace boost {

#ifndef BOOST_IS_CONVERTIBLE


namespace detail {

#if !defined(BOOST_NO_SFINAE_EXPR) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && !(defined(BOOST_GCC) && (BOOST_GCC < 40700))


#  define BOOST_TT_CXX11_IS_CONVERTIBLE

template <class A, class B, class C>
struct or_helper
{
static const bool value = (A::value || B::value || C::value);
};

template<typename From, typename To, bool b = or_helper<boost::is_void<From>, boost::is_function<To>, boost::is_array<To> >::value>
struct is_convertible_basic_impl
{
static const bool value = is_void<To>::value; 
};

template<typename From, typename To>
class is_convertible_basic_impl<From, To, false>
{
typedef char one;
typedef int  two;

template<typename To1>
static void test_aux(To1);

template<typename From1, typename To1>
static decltype(test_aux<To1>(boost::declval<From1>()), one()) test(int);

template<typename, typename>
static two test(...);

public:
static const bool value = sizeof(test<From, To>(0)) == 1;
};

#elif defined(BOOST_BORLANDC) && (BOOST_BORLANDC < 0x560)
template <typename From, typename To>
struct is_convertible_impl
{
#pragma option push -w-8074
template <typename T> struct checker
{
static ::boost::type_traits::no_type BOOST_TT_DECL _m_check(...);
static ::boost::type_traits::yes_type BOOST_TT_DECL _m_check(T);
};

static typename add_lvalue_reference<From>::type  _m_from;
static bool const value = sizeof( checker<To>::_m_check(_m_from) )
== sizeof(::boost::type_traits::yes_type);
#pragma option pop
};

#elif defined(__GNUC__) || defined(BOOST_BORLANDC) && (BOOST_BORLANDC < 0x600)

struct any_conversion
{
template <typename T> any_conversion(const volatile T&);
template <typename T> any_conversion(const T&);
template <typename T> any_conversion(volatile T&);
template <typename T> any_conversion(T&);
};

template <typename T> struct checker
{
static boost::type_traits::no_type _m_check(any_conversion ...);
static boost::type_traits::yes_type _m_check(T, int);
};

template <typename From, typename To>
struct is_convertible_basic_impl
{
typedef typename add_lvalue_reference<From>::type lvalue_type;
typedef typename add_rvalue_reference<From>::type rvalue_type;
static lvalue_type _m_from;
#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES) && ((__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ > 6)))
static bool const value =
sizeof( boost::detail::checker<To>::_m_check(static_cast<rvalue_type>(_m_from), 0) )
== sizeof(::boost::type_traits::yes_type);
#else
static bool const value =
sizeof( boost::detail::checker<To>::_m_check(_m_from, 0) )
== sizeof(::boost::type_traits::yes_type);
#endif
};

#elif (defined(__EDG_VERSION__) && (__EDG_VERSION__ >= 245) && !defined(__ICL)) \
|| defined(__IBMCPP__) || defined(__HP_aCC)
struct any_conversion
{
template <typename T> any_conversion(const volatile T&);
template <typename T> any_conversion(const T&);
template <typename T> any_conversion(volatile T&);
template <typename T> any_conversion(T&);
};

template <typename From, typename To>
struct is_convertible_basic_impl
{
static ::boost::type_traits::no_type BOOST_TT_DECL _m_check(any_conversion ...);
static ::boost::type_traits::yes_type BOOST_TT_DECL _m_check(To, int);
typedef typename add_lvalue_reference<From>::type lvalue_type;
typedef typename add_rvalue_reference<From>::type rvalue_type; 
static lvalue_type _m_from;

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(static_cast<rvalue_type>(_m_from), 0) ) == sizeof(::boost::type_traits::yes_type)
);
#else
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(_m_from, 0) ) == sizeof(::boost::type_traits::yes_type)
);
#endif
};

#elif defined(__DMC__)

struct any_conversion
{
template <typename T> any_conversion(const volatile T&);
template <typename T> any_conversion(const T&);
template <typename T> any_conversion(volatile T&);
template <typename T> any_conversion(T&);
};

template <typename From, typename To>
struct is_convertible_basic_impl
{
template <class T>
static ::boost::type_traits::no_type BOOST_TT_DECL _m_check(any_conversion,  float, T);
static ::boost::type_traits::yes_type BOOST_TT_DECL _m_check(To, int, int);
typedef typename add_lvalue_reference<From>::type lvalue_type;
typedef typename add_rvalue_reference<From>::type rvalue_type;
static lvalue_type _m_from;

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
enum { value =
sizeof( _m_check(static_cast<rvalue_type>(_m_from), 0, 0) ) == sizeof(::boost::type_traits::yes_type)
};
#else
enum { value =
sizeof( _m_check(_m_from, 0, 0) ) == sizeof(::boost::type_traits::yes_type)
};
#endif
};

#elif defined(__MWERKS__)

template <typename From, typename To,bool FromIsFunctionRef>
struct is_convertible_basic_impl_aux;

struct any_conversion
{
template <typename T> any_conversion(const volatile T&);
template <typename T> any_conversion(const T&);
template <typename T> any_conversion(volatile T&);
template <typename T> any_conversion(T&);
};

template <typename From, typename To>
struct is_convertible_basic_impl_aux<From,To,false >
{
static ::boost::type_traits::no_type BOOST_TT_DECL _m_check(any_conversion ...);
static ::boost::type_traits::yes_type BOOST_TT_DECL _m_check(To, int);
typedef typename add_lvalue_reference<From>::type lvalue_type;
typedef typename add_rvalue_reference<From>::type rvalue_type; 
static lvalue_type _m_from;

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(static_cast<rvalue_type>(_m_from), 0) ) == sizeof(::boost::type_traits::yes_type)
);
#else
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(_m_from, 0) ) == sizeof(::boost::type_traits::yes_type)
);
#endif
};

template <typename From, typename To>
struct is_convertible_basic_impl_aux<From,To,true >
{
static ::boost::type_traits::no_type BOOST_TT_DECL _m_check(...);
static ::boost::type_traits::yes_type BOOST_TT_DECL _m_check(To);
typedef typename add_lvalue_reference<From>::type lvalue_type;
typedef typename add_rvalue_reference<From>::type rvalue_type;
static lvalue_type _m_from;
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(static_cast<rvalue_type>(_m_from)) ) == sizeof(::boost::type_traits::yes_type)
);
#else
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(_m_from) ) == sizeof(::boost::type_traits::yes_type)
);
#endif
};

template <typename From, typename To>
struct is_convertible_basic_impl:
is_convertible_basic_impl_aux<
From,To,
::boost::is_function<typename ::boost::remove_reference<From>::type>::value
>
{};

#else

template <typename From>
struct is_convertible_basic_impl_add_lvalue_reference
: add_lvalue_reference<From>
{};

template <typename From>
struct is_convertible_basic_impl_add_lvalue_reference<From[]>
{
typedef From type [];
};

template <typename From, typename To>
struct is_convertible_basic_impl
{
static ::boost::type_traits::no_type BOOST_TT_DECL _m_check(...);
static ::boost::type_traits::yes_type BOOST_TT_DECL _m_check(To);
typedef typename is_convertible_basic_impl_add_lvalue_reference<From>::type lvalue_type;
static lvalue_type _m_from;
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4244)
#if BOOST_WORKAROUND(BOOST_MSVC_FULL_VER, >= 140050000)
#pragma warning(disable:6334)
#endif
#endif
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
typedef typename add_rvalue_reference<From>::type rvalue_type; 
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(static_cast<rvalue_type>(_m_from)) ) == sizeof(::boost::type_traits::yes_type)
);
#else
BOOST_STATIC_CONSTANT(bool, value =
sizeof( _m_check(_m_from) ) == sizeof(::boost::type_traits::yes_type)
);
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
};

#endif 

#if defined(__DMC__)
template <typename From, typename To>
struct is_convertible_impl
{
enum { 
value = ( ::boost::detail::is_convertible_basic_impl<From,To>::value && ! ::boost::is_array<To>::value && ! ::boost::is_function<To>::value) 
};
};
#elif !defined(BOOST_BORLANDC) || BOOST_BORLANDC > 0x551
template <typename From, typename To>
struct is_convertible_impl
{
BOOST_STATIC_CONSTANT(bool, value = ( ::boost::detail::is_convertible_basic_impl<From, To>::value && !::boost::is_array<To>::value && !::boost::is_function<To>::value));
};
#endif

template <bool trivial1, bool trivial2, bool abstract_target>
struct is_convertible_impl_select
{
template <class From, class To>
struct rebind
{
typedef is_convertible_impl<From, To> type;
};
};

template <>
struct is_convertible_impl_select<true, true, false>
{
template <class From, class To>
struct rebind
{
typedef true_type type;
};
};

template <>
struct is_convertible_impl_select<false, false, true>
{
template <class From, class To>
struct rebind
{
typedef false_type type;
};
};

template <>
struct is_convertible_impl_select<true, false, true>
{
template <class From, class To>
struct rebind
{
typedef false_type type;
};
};

template <typename From, typename To>
struct is_convertible_impl_dispatch_base
{
#if !BOOST_WORKAROUND(__HP_aCC, < 60700)
typedef is_convertible_impl_select< 
::boost::is_arithmetic<From>::value, 
::boost::is_arithmetic<To>::value,
#if !defined(BOOST_NO_IS_ABSTRACT) && !defined(BOOST_TT_CXX11_IS_CONVERTIBLE)
::boost::is_abstract<To>::value
#else
false
#endif
> selector;
#else
typedef is_convertible_impl_select<false, false, false> selector;
#endif
typedef typename selector::template rebind<From, To> isc_binder;
typedef typename isc_binder::type type;
};

template <typename From, typename To>
struct is_convertible_impl_dispatch 
: public is_convertible_impl_dispatch_base<From, To>::type
{};

#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS

template <> struct is_convertible_impl_dispatch<void, void> : public true_type{};
template <> struct is_convertible_impl_dispatch<void, void const> : public true_type{};
template <> struct is_convertible_impl_dispatch<void, void const volatile> : public true_type{};
template <> struct is_convertible_impl_dispatch<void, void volatile> : public true_type{};

template <> struct is_convertible_impl_dispatch<void const, void> : public true_type{};
template <> struct is_convertible_impl_dispatch<void const, void const> : public true_type{};
template <> struct is_convertible_impl_dispatch<void const, void const volatile> : public true_type{};
template <> struct is_convertible_impl_dispatch<void const, void volatile> : public true_type{};

template <> struct is_convertible_impl_dispatch<void const volatile, void> : public true_type{};
template <> struct is_convertible_impl_dispatch<void const volatile, void const> : public true_type{};
template <> struct is_convertible_impl_dispatch<void const volatile, void const volatile> : public true_type{};
template <> struct is_convertible_impl_dispatch<void const volatile, void volatile> : public true_type{};

template <> struct is_convertible_impl_dispatch<void volatile, void> : public true_type{};
template <> struct is_convertible_impl_dispatch<void volatile, void const> : public true_type{};
template <> struct is_convertible_impl_dispatch<void volatile, void const volatile> : public true_type{};
template <> struct is_convertible_impl_dispatch<void volatile, void volatile> : public true_type{};

#else
template <> struct is_convertible_impl_dispatch<void, void> : public true_type{};
#endif 

template <class To> struct is_convertible_impl_dispatch<void, To> : public false_type{};
template <class From> struct is_convertible_impl_dispatch<From, void> : public false_type{};

#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <class To> struct is_convertible_impl_dispatch<void const, To> : public false_type{};
template <class From> struct is_convertible_impl_dispatch<From, void const> : public false_type{};
template <class To> struct is_convertible_impl_dispatch<void const volatile, To> : public false_type{};
template <class From> struct is_convertible_impl_dispatch<From, void const volatile> : public false_type{};
template <class To> struct is_convertible_impl_dispatch<void volatile, To> : public false_type{};
template <class From> struct is_convertible_impl_dispatch<From, void volatile> : public false_type{};
#endif

} 

template <class From, class To> 
struct is_convertible : public integral_constant<bool, ::boost::detail::is_convertible_impl_dispatch<From, To>::value> 
{
BOOST_STATIC_ASSERT_MSG(boost::is_complete<To>::value || boost::is_void<To>::value || boost::is_array<To>::value, "Destination argument type to is_convertible must be a complete type");
BOOST_STATIC_ASSERT_MSG(boost::is_complete<From>::value || boost::is_void<From>::value || boost::is_array<From>::value, "From argument type to is_convertible must be a complete type");
};

#else

template <class From, class To>
struct is_convertible : public integral_constant<bool, BOOST_IS_CONVERTIBLE(From, To)> 
{
#if BOOST_WORKAROUND(BOOST_MSVC, <= 1900)
BOOST_STATIC_ASSERT_MSG(boost::is_complete<From>::value || boost::is_void<From>::value || boost::is_array<From>::value || boost::is_reference<From>::value, "From argument type to is_convertible must be a complete type");
#endif
#if defined(__clang__)
BOOST_STATIC_ASSERT_MSG(boost::is_complete<To>::value || boost::is_void<To>::value || boost::is_array<To>::value, "Destination argument type to is_convertible must be a complete type");
BOOST_STATIC_ASSERT_MSG(boost::is_complete<From>::value || boost::is_void<From>::value || boost::is_array<From>::value, "From argument type to is_convertible must be a complete type");
#endif
};

#endif

} 

#endif 
