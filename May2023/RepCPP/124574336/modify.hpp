
#ifndef BOOST_SPIRIT_MODIFY_OCTOBER_25_2008_0142PM
#define BOOST_SPIRIT_MODIFY_OCTOBER_25_2008_0142PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/type_traits/is_base_of.hpp>
#include <boost/spirit/home/support/unused.hpp>

namespace boost { namespace spirit
{
template <typename Domain, typename T, typename Enable = void>
struct is_modifier_directive;

template <typename Modifiers, typename T>
struct has_modifier
: is_base_of<T, Modifiers> {};

template <typename Current, typename New, typename Enable = void>
struct compound_modifier : Current, New
{
compound_modifier()
: Current(), New() {}

compound_modifier(Current const& current, New const& new_)
: Current(current), New(new_) {}
};

template <typename Current, typename New>
struct compound_modifier<
Current, New, typename enable_if<has_modifier<Current, New> >::type>
: Current
{
compound_modifier()
: Current() {}

compound_modifier(Current const& current, New const&)
: Current(current) {}
};

template <typename New, typename Enable>
struct compound_modifier<unused_type, New, Enable> : New
{
compound_modifier()
: New() {}

compound_modifier(unused_type, New const& new_)
: New(new_) {}
};

template <typename Domain, typename Enable = void>
struct modify
{
typedef void proto_is_callable_;

template <typename Sig>
struct result;

template <typename This, typename Tag, typename Modifiers>
struct result<This(Tag, Modifiers)>
{
typedef typename remove_const<
typename remove_reference<Tag>::type>::type
tag_type;
typedef typename remove_const<
typename remove_reference<Modifiers>::type>::type
modifiers_type;

typedef typename mpl::if_<
is_modifier_directive<Domain, tag_type>
, compound_modifier<modifiers_type, tag_type>
, Modifiers>::type
type;
};

template <typename Tag, typename Modifiers>
typename result<modify(Tag, Modifiers)>::type
operator()(Tag tag, Modifiers modifiers) const
{
return op(tag, modifiers, is_modifier_directive<Domain, Tag>());
}

template <typename Tag, typename Modifiers>
Modifiers
op(Tag , Modifiers modifiers, mpl::false_) const
{
return modifiers;
}

template <typename Tag, typename Modifiers>
compound_modifier<Modifiers, Tag>
op(Tag tag, Modifiers modifiers, mpl::true_) const
{
return compound_modifier<Modifiers, Tag>(modifiers, tag);
}
};
}}

#endif
