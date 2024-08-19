
#include <boost/config.hpp>
#include <boost/type_traits/detail/config.hpp>

#if defined(BOOST_TT_HAS_ACCURATE_BINARY_OPERATOR_DETECTION)

#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/make_void.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <utility>

#ifdef BOOST_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#endif
#if defined(BOOST_MSVC)
#   pragma warning ( push )
#   pragma warning ( disable : 4804)
#endif

namespace boost
{

namespace binary_op_detail {

struct dont_care;

template <class T, class Ret, class = void>
struct BOOST_JOIN(BOOST_TT_TRAIT_NAME, _ret_imp) : public boost::false_type {};

template <class T, class Ret>
struct BOOST_JOIN(BOOST_TT_TRAIT_NAME, _ret_imp)<T, Ret, typename boost::make_void<decltype(BOOST_TT_TRAIT_OP std::declval<typename add_reference<T>::type>()) >::type>
: public boost::integral_constant<bool, ::boost::is_convertible<decltype(BOOST_TT_TRAIT_OP std::declval<typename add_reference<T>::type>() ), Ret>::value> {};

template <class T, class = void >
struct BOOST_JOIN(BOOST_TT_TRAIT_NAME, _void_imp) : public boost::false_type {};

template <class T>
struct BOOST_JOIN(BOOST_TT_TRAIT_NAME, _void_imp)<T, typename boost::make_void<decltype(BOOST_TT_TRAIT_OP std::declval<typename add_reference<T>::type>())>::type>
: public boost::integral_constant<bool, ::boost::is_void<decltype(BOOST_TT_TRAIT_OP std::declval<typename add_reference<T>::type>())>::value> {};

template <class T, class = void>
struct BOOST_JOIN(BOOST_TT_TRAIT_NAME, _dc_imp) : public boost::false_type {};

template <class T>
struct BOOST_JOIN(BOOST_TT_TRAIT_NAME, _dc_imp)<T, typename boost::make_void<decltype(BOOST_TT_TRAIT_OP std::declval<typename add_reference<T>::type>() )>::type>
: public boost::true_type {};

}

template <class T, class Ret = boost::binary_op_detail::dont_care>
struct BOOST_TT_TRAIT_NAME : public boost::binary_op_detail::BOOST_JOIN(BOOST_TT_TRAIT_NAME, _ret_imp) <T, Ret> {};
template <class T>
struct BOOST_TT_TRAIT_NAME<T, void> : public boost::binary_op_detail::BOOST_JOIN(BOOST_TT_TRAIT_NAME, _void_imp) <T> {};
template <class T>
struct BOOST_TT_TRAIT_NAME<T, boost::binary_op_detail::dont_care> : public boost::binary_op_detail::BOOST_JOIN(BOOST_TT_TRAIT_NAME, _dc_imp) <T> {};


}

#ifdef BOOST_GCC
#pragma GCC diagnostic pop
#endif
#if defined(BOOST_MSVC)
#   pragma warning ( pop )
#endif

#else

#include <boost/type_traits/detail/yes_no_type.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_void.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>

#if defined(__GNUC__)
#   pragma GCC system_header
#elif defined(BOOST_MSVC)
#   pragma warning ( push )
#   pragma warning ( disable : 4146 4804 4913 4244 4800)
#   if BOOST_WORKAROUND(BOOST_MSVC_FULL_VER, >= 140050000)
#       pragma warning ( disable : 6334)
#   endif
#   if BOOST_WORKAROUND(_MSC_VER, >= 1913)
#       pragma warning ( disable : 4834)
#   endif
#endif



namespace boost {
namespace detail {

namespace BOOST_JOIN(BOOST_TT_TRAIT_NAME,_impl) {

template <typename T> T &make();



struct no_operator { };

struct any { template <class T> any(T const&); };

no_operator operator BOOST_TT_TRAIT_OP (const any&);



struct returns_void_t { };
template <typename T> int operator,(const T&, returns_void_t);
template <typename T> int operator,(const volatile T&, returns_void_t);

template < typename Rhs >
struct operator_returns_void {
static ::boost::type_traits::yes_type returns_void(returns_void_t);
static ::boost::type_traits::no_type returns_void(int);
BOOST_STATIC_CONSTANT(bool, value = (sizeof(::boost::type_traits::yes_type)==sizeof(returns_void((BOOST_TT_TRAIT_OP make<Rhs>(),returns_void_t())))));
};



struct dont_care { };

template < typename Rhs, typename Ret, bool Returns_void >
struct operator_returns_Ret;

template < typename Rhs >
struct operator_returns_Ret < Rhs, dont_care, true > {
BOOST_STATIC_CONSTANT(bool, value = true);
};

template < typename Rhs >
struct operator_returns_Ret < Rhs, dont_care, false > {
BOOST_STATIC_CONSTANT(bool, value = true);
};

template < typename Rhs >
struct operator_returns_Ret < Rhs, void, true > {
BOOST_STATIC_CONSTANT(bool, value = true);
};

template < typename Rhs >
struct operator_returns_Ret < Rhs, void, false > {
BOOST_STATIC_CONSTANT(bool, value = false);
};

template < typename Rhs, typename Ret >
struct operator_returns_Ret < Rhs, Ret, true > {
BOOST_STATIC_CONSTANT(bool, value = false);
};

template < typename Rhs, typename Ret >
struct operator_returns_Ret < Rhs, Ret, false > {
static ::boost::type_traits::yes_type is_convertible_to_Ret(Ret); 
static ::boost::type_traits::no_type is_convertible_to_Ret(...); 

BOOST_STATIC_CONSTANT(bool, value = (sizeof(is_convertible_to_Ret(BOOST_TT_TRAIT_OP make<Rhs>()))==sizeof(::boost::type_traits::yes_type)));
};



struct has_operator { };
no_operator operator,(no_operator, has_operator);

template < typename Rhs >
struct operator_exists {
static ::boost::type_traits::yes_type s_check(has_operator); 
static ::boost::type_traits::no_type s_check(no_operator); 

BOOST_STATIC_CONSTANT(bool, value = (sizeof(s_check(((BOOST_TT_TRAIT_OP make<Rhs>()),make<has_operator>())))==sizeof(::boost::type_traits::yes_type)));
};


template < typename Rhs, typename Ret, bool Forbidden_if >
struct trait_impl1;

template < typename Rhs, typename Ret >
struct trait_impl1 < Rhs, Ret, true > {
BOOST_STATIC_CONSTANT(bool, value = false);
};

template < typename Rhs, typename Ret >
struct trait_impl1 < Rhs, Ret, false > {
BOOST_STATIC_CONSTANT(bool,
value = (operator_exists < Rhs >::value && operator_returns_Ret < Rhs, Ret, operator_returns_void < Rhs >::value >::value));
};

template < typename Ret >
struct trait_impl1 < void, Ret, false > {
BOOST_STATIC_CONSTANT(bool, value = false);
};

template < typename Rhs, typename Ret >
struct trait_impl {
typedef typename ::boost::remove_reference<Rhs>::type Rhs_noref;
typedef typename ::boost::remove_cv<Rhs_noref>::type Rhs_nocv;
typedef typename ::boost::remove_cv< typename ::boost::remove_reference< typename ::boost::remove_pointer<Rhs_noref>::type >::type >::type Rhs_noptr;
BOOST_STATIC_CONSTANT(bool, value = (trait_impl1 < Rhs_noref, Ret, BOOST_TT_FORBIDDEN_IF >::value));
};

} 
} 

template <class Rhs, class Ret=::boost::detail::BOOST_JOIN(BOOST_TT_TRAIT_NAME,_impl)::dont_care>
struct BOOST_TT_TRAIT_NAME : public integral_constant<bool, (::boost::detail::BOOST_JOIN(BOOST_TT_TRAIT_NAME, _impl)::trait_impl < Rhs, Ret >::value)>{};

} 

#if defined(BOOST_MSVC)
#   pragma warning ( pop )
#endif

#endif

