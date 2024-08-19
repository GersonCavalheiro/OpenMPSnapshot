
#ifndef BOOST_SPIRIT_QI_DETAIL_PASS_CONTAINER_HPP
#define BOOST_SPIRIT_QI_DETAIL_PASS_CONTAINER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/container.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Sequence, typename Attribute, typename ValueType>
struct negate_weak_substitute_if_not
: mpl::if_<
Sequence
, typename traits::is_weak_substitute<Attribute, ValueType>::type
, typename mpl::not_<
traits::is_weak_substitute<Attribute, ValueType>
>::type>
{};


template <typename Container, typename ValueType, typename Attribute
, typename Sequence, typename Enable = void>
struct pass_through_container_base
: negate_weak_substitute_if_not<Sequence, Attribute, ValueType>
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence = mpl::true_>
struct not_compatible_element
: mpl::and_<
negate_weak_substitute_if_not<Sequence, Attribute, Container>
, negate_weak_substitute_if_not<Sequence, Attribute, ValueType> >
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence
, bool IsSequence = fusion::traits::is_sequence<ValueType>::value>
struct pass_through_container_fusion_sequence
{
typedef typename mpl::find_if<
Attribute, not_compatible_element<Container, ValueType, mpl::_1>
>::type iter;
typedef typename mpl::end<Attribute>::type end;

typedef typename is_same<iter, end>::type type;
};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container_fusion_sequence<
Container, ValueType, Attribute, Sequence, true>
{
typedef typename mpl::find_if<
Attribute
, not_compatible_element<Container, ValueType, mpl::_1, Sequence>
>::type iter;
typedef typename mpl::end<Attribute>::type end;

typedef typename is_same<iter, end>::type type;
};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container_base<Container, ValueType, Attribute
, Sequence
, typename enable_if<fusion::traits::is_sequence<Attribute> >::type>
: pass_through_container_fusion_sequence<
Container, ValueType, Attribute, Sequence>
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence, typename AttributeValueType
, bool IsSequence = fusion::traits::is_sequence<AttributeValueType>::value>
struct pass_through_container_container
: mpl::or_<
traits::is_weak_substitute<Attribute, Container>
, traits::is_weak_substitute<AttributeValueType, Container> >
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence, typename AttributeValueType>
struct pass_through_container_container<
Container, ValueType, Attribute, Sequence, AttributeValueType, true>
: pass_through_container_fusion_sequence<
Container, ValueType, AttributeValueType, Sequence>
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container_base<
Container, ValueType, Attribute, Sequence
, typename enable_if<traits::is_container<Attribute> >::type>
: detail::pass_through_container_container<
Container, ValueType, Attribute, Sequence
, typename traits::container_value<Attribute>::type>
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence
, bool IsSequence = fusion::traits::is_sequence<Attribute>::value>
struct pass_through_container_optional
: mpl::or_<
traits::is_weak_substitute<Attribute, Container>
, traits::is_weak_substitute<Attribute, ValueType> >
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container_optional<
Container, ValueType, Attribute, Sequence, true>
: pass_through_container_fusion_sequence<
Container, ValueType, Attribute, Sequence>
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container
: pass_through_container_base<Container, ValueType, Attribute, Sequence>
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container<
Container, ValueType, boost::optional<Attribute>, Sequence>
: pass_through_container_optional<
Container, ValueType, Attribute, Sequence>
{};

template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container<
Container, boost::optional<ValueType>, boost::optional<Attribute>
, Sequence>
: mpl::not_<traits::is_weak_substitute<Attribute, ValueType> >
{};


#if !defined(BOOST_VARIANT_DO_NOT_USE_VARIADIC_TEMPLATES)
template <typename Container, typename ValueType, typename Sequence
, typename T>
struct pass_through_container<Container, ValueType, boost::variant<T>
, Sequence>
: pass_through_container<Container, ValueType, T, Sequence>
{};

template <typename Container, typename ValueType, typename Sequence
, typename T0, typename ...TN>
struct pass_through_container<Container, ValueType
, boost::variant<T0, TN...>, Sequence>
: mpl::bool_<pass_through_container<
Container, ValueType, T0, Sequence
>::type::value || pass_through_container<
Container, ValueType, boost::variant<TN...>, Sequence
>::type::value>
{};
#else
#define BOOST_SPIRIT_PASS_THROUGH_CONTAINER(z, N, _)                          \
pass_through_container<Container, ValueType,                              \
BOOST_PP_CAT(T, N), Sequence>::type::value ||                         \


template <typename Container, typename ValueType, typename Sequence>
struct pass_through_container<Container, ValueType
, boost::detail::variant::void_, Sequence>
: mpl::false_
{};

template <typename Container, typename ValueType, typename Sequence
, BOOST_VARIANT_ENUM_PARAMS(typename T)>
struct pass_through_container<Container, ValueType
, boost::variant<BOOST_VARIANT_ENUM_PARAMS(T)>, Sequence>
: mpl::bool_<BOOST_PP_REPEAT(BOOST_VARIANT_LIMIT_TYPES
, BOOST_SPIRIT_PASS_THROUGH_CONTAINER, _) false>
{};

#undef BOOST_SPIRIT_PASS_THROUGH_CONTAINER
#endif
}}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Container, typename ValueType, typename Attribute
, typename Sequence>
struct pass_through_container<
Container, ValueType, Attribute, Sequence, qi::domain>
: qi::detail::pass_through_container<
Container, ValueType, Attribute, Sequence>
{};
}}}

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename F, typename Attr, typename Sequence>
struct pass_container
{
typedef typename F::context_type context_type;
typedef typename F::iterator_type iterator_type;

pass_container(F const& f_, Attr& attr_)
: f(f_), attr(attr_) {}

template <typename Component>
bool dispatch_container(Component const& component, mpl::false_) const
{
typename traits::container_value<Attr>::type val =
typename traits::container_value<Attr>::type();

iterator_type save = f.first;
bool r = f(component, val);
if (!r)
{
r = !traits::push_back(attr, val);
if (r)
f.first = save;
}
return r;
}

template <typename Component>
bool dispatch_container(Component const& component, mpl::true_) const
{
return f(component, attr);
}

template <typename Component>
bool dispatch_attribute(Component const& component, mpl::false_) const
{
return f(component, unused);
}

template <typename Component>
bool dispatch_attribute(Component const& component, mpl::true_) const
{
typedef typename traits::container_value<Attr>::type value_type;
typedef typename traits::attribute_of<
Component, context_type, iterator_type>::type
rhs_attribute;

typedef mpl::and_<
traits::handles_container<
Component, Attr, context_type, iterator_type>
, traits::pass_through_container<
Attr, value_type, rhs_attribute, Sequence, qi::domain>
> predicate;

return dispatch_container(component, predicate());
}

template <typename Component>
bool operator()(Component const& component) const
{
typedef typename traits::not_is_unused<
typename traits::attribute_of<
Component, context_type, iterator_type
>::type
>::type predicate;

traits::make_container(attr);

return dispatch_attribute(component, predicate());
}

F f;
Attr& attr;

BOOST_DELETED_FUNCTION(pass_container& operator= (pass_container const&))
};

template <typename F, typename Attr>
inline pass_container<F, Attr, mpl::false_>
make_pass_container(F const& f, Attr& attr)
{
return pass_container<F, Attr, mpl::false_>(f, attr);
}

template <typename F, typename Attr>
inline pass_container<F, Attr, mpl::true_>
make_sequence_pass_container(F const& f, Attr& attr)
{
return pass_container<F, Attr, mpl::true_>(f, attr);
}
}}}}

#endif

