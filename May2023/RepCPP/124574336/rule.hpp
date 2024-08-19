
#if !defined(BOOST_SPIRIT_RULE_FEBRUARY_12_2007_1020AM)
#define BOOST_SPIRIT_RULE_FEBRUARY_12_2007_1020AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/config.hpp>
#include <boost/function.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/cons.hpp>
#include <boost/fusion/include/as_list.hpp>
#include <boost/fusion/include/as_vector.hpp>

#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/argument.hpp>
#include <boost/spirit/home/support/context.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/nonterminal/extract_param.hpp>
#include <boost/spirit/home/support/nonterminal/locals.hpp>
#include <boost/spirit/home/qi/reference.hpp>
#include <boost/spirit/home/qi/nonterminal/detail/parameterized.hpp>
#include <boost/spirit/home/qi/nonterminal/detail/parser_binder.hpp>
#include <boost/spirit/home/qi/nonterminal/nonterminal_fwd.hpp>
#include <boost/spirit/home/qi/skip_over.hpp>

#include <boost/proto/extends.hpp>
#include <boost/proto/traits.hpp>
#include <boost/type_traits/is_reference.hpp>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable: 4355) 
# pragma warning(disable: 4127) 
#endif

namespace boost { namespace spirit { namespace qi
{
BOOST_PP_REPEAT(SPIRIT_ATTRIBUTES_LIMIT, SPIRIT_USING_ATTRIBUTE, _)

using spirit::_pass_type;
using spirit::_val_type;
using spirit::_a_type;
using spirit::_b_type;
using spirit::_c_type;
using spirit::_d_type;
using spirit::_e_type;
using spirit::_f_type;
using spirit::_g_type;
using spirit::_h_type;
using spirit::_i_type;
using spirit::_j_type;

#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS

using spirit::_pass;
using spirit::_val;
using spirit::_a;
using spirit::_b;
using spirit::_c;
using spirit::_d;
using spirit::_e;
using spirit::_f;
using spirit::_g;
using spirit::_h;
using spirit::_i;
using spirit::_j;

#endif

using spirit::info;
using spirit::locals;

template <
typename Iterator, typename T1, typename T2, typename T3
, typename T4>
struct rule
: proto::extends<
typename proto::terminal<
reference<rule<Iterator, T1, T2, T3, T4> const>
>::type
, rule<Iterator, T1, T2, T3, T4>
>
, parser<rule<Iterator, T1, T2, T3, T4> >
{
typedef Iterator iterator_type;
typedef rule<Iterator, T1, T2, T3, T4> this_type;
typedef reference<this_type const> reference_;
typedef typename proto::terminal<reference_>::type terminal;
typedef proto::extends<terminal, this_type> base_type;
typedef mpl::vector<T1, T2, T3, T4> template_params;

typedef typename
spirit::detail::extract_locals<template_params>::type
locals_type;

typedef typename
spirit::detail::extract_component<
qi::domain, template_params>::type
skipper_type;

typedef typename
spirit::detail::extract_encoding<template_params>::type
encoding_type;

typedef typename
spirit::detail::extract_sig<template_params, encoding_type, qi::domain>::type
sig_type;

typedef typename
spirit::detail::attr_from_sig<sig_type>::type
attr_type;
BOOST_STATIC_ASSERT_MSG(
!is_reference<attr_type>::value,
"Reference qualifier on Qi rule attribute is meaningless");
typedef attr_type& attr_reference_type;

typedef typename
spirit::detail::params_from_sig<sig_type>::type
parameter_types;

static size_t const params_size =
fusion::result_of::size<parameter_types>::type::value;

typedef context<
fusion::cons<attr_reference_type, parameter_types>
, locals_type>
context_type;

typedef function<
bool(Iterator& first, Iterator const& last
, context_type& context
, skipper_type const& skipper
)>
function_type;

typedef typename
mpl::if_<
is_same<encoding_type, unused_type>
, unused_type
, tag::char_code<tag::encoding, encoding_type>
>::type
encoding_modifier_type;

explicit rule(std::string const& name = "unnamed-rule")
: base_type(terminal::make(reference_(*this)))
, name_(name)
{
}

rule(rule const& rhs)
: base_type(terminal::make(reference_(*this)))
, name_(rhs.name_)
, f(rhs.f)
{
}

template <typename Auto, typename Expr>
static void define(rule& , Expr const& , mpl::false_)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Expr);
}

template <typename Auto, typename Expr>
static void define(rule& lhs, Expr const& expr, mpl::true_)
{
lhs.f = detail::bind_parser<Auto>(
compile<qi::domain>(expr, encoding_modifier_type()));
}

template <typename Expr>
rule(Expr const& expr, std::string const& name = "unnamed-rule")
: base_type(terminal::make(reference_(*this)))
, name_(name)
{
define<mpl::false_>(*this, expr, traits::matches<qi::domain, Expr>());
}

rule& operator=(rule const& rhs)
{
BOOST_ASSERT(rhs.f && "Did you mean rhs.alias() instead of rhs?");

f = rhs.f;
name_ = rhs.name_;
return *this;
}

std::string const& name() const
{
return name_;
}

void name(std::string const& str)
{
name_ = str;
}

template <typename Expr>
rule& operator=(Expr const& expr)
{
define<mpl::false_>(*this, expr, traits::matches<qi::domain, Expr>());
return *this;
}

#if !BOOST_WORKAROUND(BOOST_MSVC, < 1400)
template <typename Expr>
friend rule& operator%=(rule& r, Expr const& expr)
{
define<mpl::true_>(r, expr, traits::matches<qi::domain, Expr>());
return r;
}

#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
template <typename Expr>
friend rule& operator%=(rule& r, Expr& expr)
{
return r %= static_cast<Expr const&>(expr);
}
#else
template <typename Expr>
friend rule& operator%=(rule& r, Expr&& expr)
{
define<mpl::true_>(r, expr, traits::matches<qi::domain, Expr>());
return r;
}
#endif

#else
template <typename OutputIterator_, typename T1_, typename T2_
, typename T3_, typename T4_, typename Expr>
friend rule<OutputIterator_, T1_, T2_, T3_, T4_>& operator%=(
rule<OutputIterator_, T1_, T2_, T3_, T4_>& r, Expr const& expr);

template <typename OutputIterator_, typename T1_, typename T2_
, typename T3_, typename T4_, typename Expr>
friend rule<OutputIterator_, T1_, T2_, T3_, T4_>& operator%=(
rule<OutputIterator_, T1_, T2_, T3_, T4_>& r, Expr& expr);
#endif

template <typename Context, typename Iterator_>
struct attribute
{
typedef attr_type type;
};

template <typename Context, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& , Skipper const& skipper
, Attribute& attr_param) const
{
BOOST_STATIC_ASSERT_MSG((is_same<skipper_type, unused_type>::value ||
!is_same<Skipper, unused_type>::value),
"The rule was instantiated with a skipper type but you have not pass any. "
"Did you use `parse` instead of `phrase_parse`?");
BOOST_STATIC_ASSERT_MSG(
(is_convertible<Skipper const&, skipper_type>::value),
"The passed skipper is not compatible/convertible to one "
"that the rule was instantiated with");
if (f)
{
if (is_same<skipper_type, unused_type>::value)
qi::skip_over(first, last, skipper);

typedef traits::transform_attribute<
Attribute, attr_type, domain>
transform;

typename transform::type attr_ = transform::pre(attr_param);

context_type context(attr_);

if (f(first, last, context, skipper))
{
transform::post(attr_param, attr_);
return true;
}

transform::fail(attr_param);
}
return false;
}

template <typename Context, typename Skipper
, typename Attribute, typename Params>
bool parse(Iterator& first, Iterator const& last
, Context& caller_context, Skipper const& skipper
, Attribute& attr_param, Params const& params) const
{
BOOST_STATIC_ASSERT_MSG((is_same<skipper_type, unused_type>::value ||
!is_same<Skipper, unused_type>::value),
"The rule was instantiated with a skipper type but you have not pass any. "
"Did you use `parse` instead of `phrase_parse`?");
BOOST_STATIC_ASSERT_MSG(
(is_convertible<Skipper const&, skipper_type>::value),
"The passed skipper is not compatible/convertible to one "
"that the rule was instantiated with");
if (f)
{
if (is_same<skipper_type, unused_type>::value)
qi::skip_over(first, last, skipper);

typedef traits::transform_attribute<
Attribute, attr_type, domain>
transform;

typename transform::type attr_ = transform::pre(attr_param);

context_type context(attr_, params, caller_context);

if (f(first, last, context, skipper))
{
transform::post(attr_param, attr_);
return true;
}

transform::fail(attr_param);
}
return false;
}

template <typename Context>
info what(Context& ) const
{
return info(name_);
}

reference_ alias() const
{
return reference_(*this);
}

typename proto::terminal<this_type>::type copy() const
{
typename proto::terminal<this_type>::type result = {*this};
return result;
}

rule const& get_parameterized_subject() const { return *this; }
typedef rule parameterized_subject_type;
#include <boost/spirit/home/qi/nonterminal/detail/fcall.hpp>

std::string name_;
function_type f;
};

#if BOOST_WORKAROUND(BOOST_MSVC, < 1400)
template <typename OutputIterator_, typename T1_, typename T2_
, typename T3_, typename T4_, typename Expr>
rule<OutputIterator_, T1_, T2_, T3_, T4_>& operator%=(
rule<OutputIterator_, T1_, T2_, T3_, T4_>& r, Expr const& expr)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Expr);

typedef typename
rule<OutputIterator_, T1_, T2_, T3_, T4_>::encoding_modifier_type
encoding_modifier_type;

r.f = detail::bind_parser<mpl::true_>(
compile<qi::domain>(expr, encoding_modifier_type()));
return r;
}

template <typename Iterator_, typename T1_, typename T2_
, typename T3_, typename T4_, typename Expr>
rule<Iterator_, T1_, T2_, T3_, T4_>& operator%=(
rule<Iterator_, T1_, T2_, T3_, T4_>& r, Expr& expr)
{
return r %= static_cast<Expr const&>(expr);
}
#endif
}}}

namespace boost { namespace spirit { namespace traits
{
template <
typename IteratorA, typename IteratorB, typename Attribute
, typename Context, typename T1, typename T2, typename T3, typename T4>
struct handles_container<
qi::rule<IteratorA, T1, T2, T3, T4>, Attribute, Context, IteratorB>
: traits::is_container<
typename attribute_of<
qi::rule<IteratorA, T1, T2, T3, T4>, Context, IteratorB
>::type
>
{};
}}}

#if defined(BOOST_MSVC)
# pragma warning(pop)
#endif

#endif
