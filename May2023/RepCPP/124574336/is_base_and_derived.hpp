

#ifndef BOOST_TT_IS_BASE_AND_DERIVED_HPP_INCLUDED
#define BOOST_TT_IS_BASE_AND_DERIVED_HPP_INCLUDED

#include <boost/type_traits/intrinsics.hpp>
#include <boost/type_traits/integral_constant.hpp>
#ifndef BOOST_IS_BASE_OF
#include <boost/type_traits/is_class.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#endif
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost {

namespace detail {

#ifndef BOOST_IS_BASE_OF
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x581)) \
&& !BOOST_WORKAROUND(__SUNPRO_CC , <= 0x540) \
&& !BOOST_WORKAROUND(__EDG_VERSION__, <= 243) \
&& !BOOST_WORKAROUND(__DMC__, BOOST_TESTED_AT(0x840))




template <typename B, typename D>
struct bd_helper
{
#if !BOOST_WORKAROUND(BOOST_MSVC, == 1310)
template <typename T>
static type_traits::yes_type check_sig(D const volatile *, T);
static type_traits::no_type  check_sig(B const volatile *, int);
#else
static type_traits::yes_type check_sig(D const volatile *, long);
static type_traits::no_type  check_sig(B const volatile * const&, int);
#endif
};

template<typename B, typename D>
struct is_base_and_derived_impl2
{
#if BOOST_WORKAROUND(BOOST_MSVC_FULL_VER, >= 140050000)
#pragma warning(push)
#pragma warning(disable:6334)
#endif
BOOST_STATIC_ASSERT(sizeof(B) != 0);
BOOST_STATIC_ASSERT(sizeof(D) != 0);

struct Host
{
#if !BOOST_WORKAROUND(BOOST_MSVC, == 1310)
operator B const volatile *() const;
#else
operator B const volatile * const&() const;
#endif
operator D const volatile *();
};

BOOST_STATIC_CONSTANT(bool, value =
sizeof(bd_helper<B,D>::check_sig(Host(), 0)) == sizeof(type_traits::yes_type));
#if BOOST_WORKAROUND(BOOST_MSVC_FULL_VER, >= 140050000)
#pragma warning(pop)
#endif
};

#else

template<typename B, typename D>
struct is_base_and_derived_impl2
{
BOOST_STATIC_CONSTANT(bool, value =
(::boost::is_convertible<D*,B*>::value));
};

#define BOOST_BROKEN_IS_BASE_AND_DERIVED

#endif

template <typename B, typename D>
struct is_base_and_derived_impl3
{
BOOST_STATIC_CONSTANT(bool, value = false);
};

template <bool ic1, bool ic2, bool iss>
struct is_base_and_derived_select
{
template <class T, class U>
struct rebind
{
typedef is_base_and_derived_impl3<T,U> type;
};
};

template <>
struct is_base_and_derived_select<true,true,false>
{
template <class T, class U>
struct rebind
{
typedef is_base_and_derived_impl2<T,U> type;
};
};

template <typename B, typename D>
struct is_base_and_derived_impl
{
typedef typename remove_cv<B>::type ncvB;
typedef typename remove_cv<D>::type ncvD;

typedef is_base_and_derived_select<
::boost::is_class<B>::value,
::boost::is_class<D>::value,
::boost::is_same<ncvB,ncvD>::value> selector;
typedef typename selector::template rebind<ncvB,ncvD> binder;
typedef typename binder::type bound_type;

BOOST_STATIC_CONSTANT(bool, value = bound_type::value);
};
#else
template <typename B, typename D>
struct is_base_and_derived_impl
{
typedef typename remove_cv<B>::type ncvB;
typedef typename remove_cv<D>::type ncvD;

BOOST_STATIC_CONSTANT(bool, value = (BOOST_IS_BASE_OF(B,D) && ! ::boost::is_same<ncvB,ncvD>::value));
};
#endif
} 

template <class Base, class Derived> struct is_base_and_derived
: public integral_constant<bool, (::boost::detail::is_base_and_derived_impl<Base, Derived>::value)> {};

template <class Base, class Derived> struct is_base_and_derived<Base&, Derived> : public false_type{};
template <class Base, class Derived> struct is_base_and_derived<Base, Derived&> : public false_type{};
template <class Base, class Derived> struct is_base_and_derived<Base&, Derived&> : public false_type{};

#if BOOST_WORKAROUND(BOOST_CODEGEARC, BOOST_TESTED_AT(0x610))
template <class Base> struct is_base_and_derived<Base, Base> : public true_type{};
#endif

} 

#endif 
