
#ifndef BOOST_ASIO_EXECUTION_PREFER_ONLY_HPP
#define BOOST_ASIO_EXECUTION_PREFER_ONLY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/is_applicable_property.hpp>
#include <boost/asio/prefer.hpp>
#include <boost/asio/query.hpp>
#include <boost/asio/traits/static_query.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if defined(GENERATING_DOCUMENTATION)

namespace execution {

template <typename Property>
struct prefer_only
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_applicable_property<T, Property>::value;

static constexpr bool is_requirable = false;


static constexpr bool is_preferable = automatically_determined;

typedef typename Property::polymorphic_query_result_type
polymorphic_query_result_type;
};

} 

#else 

namespace execution {
namespace detail {

template <typename InnerProperty, typename = void>
struct prefer_only_is_preferable
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_preferable = false);
};

template <typename InnerProperty>
struct prefer_only_is_preferable<InnerProperty,
typename enable_if<
InnerProperty::is_preferable
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_preferable = true);
};

template <typename InnerProperty, typename = void>
struct prefer_only_polymorphic_query_result_type
{
};

template <typename InnerProperty>
struct prefer_only_polymorphic_query_result_type<InnerProperty,
typename void_type<
typename InnerProperty::polymorphic_query_result_type
>::type>
{
typedef typename InnerProperty::polymorphic_query_result_type
polymorphic_query_result_type;
};

template <typename InnerProperty, typename = void>
struct prefer_only_property
{
InnerProperty property;

prefer_only_property(const InnerProperty& p)
: property(p)
{
}
};

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)

template <typename InnerProperty>
struct prefer_only_property<InnerProperty,
typename void_type<
decltype(boost::asio::declval<const InnerProperty>().value())
>::type>
{
InnerProperty property;

prefer_only_property(const InnerProperty& p)
: property(p)
{
}

BOOST_ASIO_CONSTEXPR auto value() const
BOOST_ASIO_NOEXCEPT_IF((
noexcept(boost::asio::declval<const InnerProperty>().value())))
-> decltype(boost::asio::declval<const InnerProperty>().value())
{
return property.value();
}
};

#else 

struct prefer_only_memfns_base
{
void value();
};

template <typename T>
struct prefer_only_memfns_derived
: T, prefer_only_memfns_base
{
};

template <typename T, T>
struct prefer_only_memfns_check
{
};

template <typename>
char (&prefer_only_value_memfn_helper(...))[2];

template <typename T>
char prefer_only_value_memfn_helper(
prefer_only_memfns_check<
void (prefer_only_memfns_base::*)(),
&prefer_only_memfns_derived<T>::value>*);

template <typename InnerProperty>
struct prefer_only_property<InnerProperty,
typename enable_if<
sizeof(prefer_only_value_memfn_helper<InnerProperty>(0)) != 1
&& !is_same<typename InnerProperty::polymorphic_query_result_type,
void>::value
>::type>
{
InnerProperty property;

prefer_only_property(const InnerProperty& p)
: property(p)
{
}

BOOST_ASIO_CONSTEXPR typename InnerProperty::polymorphic_query_result_type
value() const
{
return property.value();
}
};

#endif 

} 

template <typename InnerProperty>
struct prefer_only :
detail::prefer_only_is_preferable<InnerProperty>,
detail::prefer_only_polymorphic_query_result_type<InnerProperty>,
detail::prefer_only_property<InnerProperty>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_requirable = false);

BOOST_ASIO_CONSTEXPR prefer_only(const InnerProperty& p)
: detail::prefer_only_property<InnerProperty>(p)
{
}

#if defined(BOOST_ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(BOOST_ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename T>
static BOOST_ASIO_CONSTEXPR
typename traits::static_query<T, InnerProperty>::result_type
static_query()
BOOST_ASIO_NOEXCEPT_IF((
traits::static_query<T, InnerProperty>::is_noexcept))
{
return traits::static_query<T, InnerProperty>::value();
}

template <typename E, typename T = decltype(prefer_only::static_query<E>())>
static BOOST_ASIO_CONSTEXPR const T static_query_v
= prefer_only::static_query<E>();
#endif 

template <typename Executor, typename Property>
friend BOOST_ASIO_CONSTEXPR
typename prefer_result<const Executor&, const InnerProperty&>::type
prefer(const Executor& ex, const prefer_only<Property>& p,
typename enable_if<
is_same<Property, InnerProperty>::value
&& can_prefer<const Executor&, const InnerProperty&>::value
>::type* = 0)
#if !defined(BOOST_ASIO_MSVC) \
&& !defined(__clang__) 
BOOST_ASIO_NOEXCEPT_IF((
is_nothrow_prefer<const Executor&, const InnerProperty&>::value))
#endif 
{
return boost::asio::prefer(ex, p.property);
}

template <typename Executor, typename Property>
friend BOOST_ASIO_CONSTEXPR
typename query_result<const Executor&, const InnerProperty&>::type
query(const Executor& ex, const prefer_only<Property>& p,
typename enable_if<
is_same<Property, InnerProperty>::value
&& can_query<const Executor&, const InnerProperty&>::value
>::type* = 0)
#if !defined(BOOST_ASIO_MSVC) \
&& !defined(__clang__) 
BOOST_ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, const InnerProperty&>::value))
#endif 
{
return boost::asio::query(ex, p.property);
}
};

#if defined(BOOST_ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(BOOST_ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename InnerProperty> template <typename E, typename T>
const T prefer_only<InnerProperty>::static_query_v;
#endif 

} 

template <typename T, typename InnerProperty>
struct is_applicable_property<T, execution::prefer_only<InnerProperty> >
: is_applicable_property<T, InnerProperty>
{
};

namespace traits {

#if !defined(BOOST_ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
|| !defined(BOOST_ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)

template <typename T, typename InnerProperty>
struct static_query<T, execution::prefer_only<InnerProperty> > :
static_query<T, const InnerProperty&>
{
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_PREFER_FREE_TRAIT)

template <typename T, typename InnerProperty>
struct prefer_free_default<T, execution::prefer_only<InnerProperty>,
typename enable_if<
can_prefer<const T&, const InnerProperty&>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_prefer<const T&, const InnerProperty&>::value));

typedef typename prefer_result<const T&,
const InnerProperty&>::type result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_QUERY_FREE_TRAIT)

template <typename T, typename InnerProperty>
struct query_free<T, execution::prefer_only<InnerProperty>,
typename enable_if<
can_query<const T&, const InnerProperty&>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<const T&, const InnerProperty&>::value));

typedef typename query_result<const T&,
const InnerProperty&>::type result_type;
};

#endif 

} 

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 