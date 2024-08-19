
#if !defined(BOOST_SPIRIT_REPOSITORY_QI_SUBRULE_AUGUST_06_2009_0239AM)
#define BOOST_SPIRIT_REPOSITORY_QI_SUBRULE_AUGUST_06_2009_0239AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/domain.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/reference.hpp>
#include <boost/spirit/home/qi/nonterminal/detail/parameterized.hpp>
#include <boost/spirit/home/qi/nonterminal/detail/parser_binder.hpp>
#include <boost/spirit/home/support/argument.hpp>
#include <boost/spirit/home/support/assert_msg.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/nonterminal/extract_param.hpp>
#include <boost/spirit/home/support/nonterminal/locals.hpp>
#include <boost/spirit/repository/home/support/subrule_context.hpp>

#include <boost/static_assert.hpp>
#include <boost/fusion/include/as_map.hpp>
#include <boost/fusion/include/at_key.hpp>
#include <boost/fusion/include/cons.hpp>
#include <boost/fusion/include/front.hpp>
#include <boost/fusion/include/has_key.hpp>
#include <boost/fusion/include/join.hpp>
#include <boost/fusion/include/make_map.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/proto/extends.hpp>
#include <boost/proto/traits.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_reference.hpp>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable: 4355) 
#endif

namespace boost { namespace spirit { namespace repository { namespace qi
{
template <typename Defs>
struct subrule_group_parser
: spirit::qi::parser<subrule_group_parser<Defs> >
{
typedef Defs defs_type;

typedef subrule_group_parser<Defs> this_type;

explicit subrule_group_parser(Defs const& defs)
: defs(defs)
{
}

template <int ID>
struct def_type
{
typedef mpl::int_<ID> id_type;

BOOST_SPIRIT_ASSERT_MSG(
(fusion::result_of::has_key<
defs_type const, id_type>::type::value)
, subrule_used_without_being_defined, (mpl::int_<ID>));

typedef typename
fusion::result_of::at_key<defs_type const, id_type>::type
type;
};

template <int ID>
typename def_type<ID>::type def() const
{
return fusion::at_key<mpl::int_<ID> >(defs);
}

template <typename Context, typename Iterator>
struct attribute
: mpl::identity<
typename remove_reference<
typename fusion::result_of::front<Defs>::type
>::type::second_type::attr_type> {};

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr) const
{
return parse_subrule(fusion::front(defs).second
, first, last, context, skipper, attr);
}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute, typename Params>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr, Params const& params) const
{
return parse_subrule(fusion::front(defs).second
, first, last, context, skipper, attr, params);
}

template <int ID, typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse_subrule_id(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr) const
{
return parse_subrule(def<ID>()
, first, last, context, skipper, attr);
}

template <int ID, typename Iterator, typename Context
, typename Skipper, typename Attribute, typename Params>
bool parse_subrule_id(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr, Params const& params) const
{
return parse_subrule(def<ID>()
, first, last, context, skipper, attr, params);
}

template <typename Def
, typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse_subrule(Def const& def
, Iterator& first, Iterator const& last
, Context& , Skipper const& skipper
, Attribute& attr) const
{
typedef typename Def::locals_type subrule_locals_type;
typedef typename Def::attr_type subrule_attr_type;
typedef typename Def::attr_reference_type subrule_attr_reference_type;
typedef typename Def::parameter_types subrule_parameter_types;

typedef
subrule_context<
this_type
, fusion::cons<
subrule_attr_reference_type, subrule_parameter_types>
, subrule_locals_type
>
context_type;

typedef traits::transform_attribute<
Attribute, subrule_attr_type, spirit::qi::domain> 
transform;

typename transform::type attr_ = transform::pre(attr);

context_type context(*this, attr_);

if (def.binder(first, last, context, skipper))
{
transform::post(attr, attr_);
return true;
}

transform::fail(attr);
return false;
}

template <typename Def
, typename Iterator, typename Context
, typename Skipper, typename Attribute, typename Params>
bool parse_subrule(Def const& def
, Iterator& first, Iterator const& last
, Context& caller_context, Skipper const& skipper
, Attribute& attr, Params const& params) const
{
typedef typename Def::locals_type subrule_locals_type;
typedef typename Def::attr_type subrule_attr_type;
typedef typename Def::attr_reference_type subrule_attr_reference_type;
typedef typename Def::parameter_types subrule_parameter_types;

typedef
subrule_context<
this_type
, fusion::cons<
subrule_attr_reference_type, subrule_parameter_types>
, subrule_locals_type
>
context_type;

typedef traits::transform_attribute<
Attribute, subrule_attr_type, spirit::qi::domain> 
transform;

typename transform::type attr_ = transform::pre(attr);

context_type context(*this, attr_, params, caller_context);

if (def.binder(first, last, context, skipper))
{
transform::post(attr, attr_);
return true;
}

transform::fail(attr);
return false;
}

template <typename Context>
info what(Context& context) const
{
return fusion::front(defs).second.binder.p.what(context);
}

Defs defs;
};

template <typename Defs>
struct subrule_group
: proto::extends<
typename proto::terminal<
subrule_group_parser<Defs>
>::type
, subrule_group<Defs>
>
{
typedef subrule_group_parser<Defs> parser_type;
typedef typename proto::terminal<parser_type>::type terminal;

static size_t const params_size =
remove_reference<
typename fusion::result_of::front<Defs>::type
>::type::second_type::params_size;

explicit subrule_group(Defs const& defs)
: subrule_group::proto_extends(terminal::make(parser_type(defs)))
{
}

parser_type const& parser() const { return proto::value(*this); }

Defs const& defs() const { return parser().defs; }

template <typename Defs2>
subrule_group<
typename fusion::result_of::as_map<
typename fusion::result_of::join<
Defs const, Defs2 const>::type>::type>
operator,(subrule_group<Defs2> const& other) const
{
typedef subrule_group<
typename fusion::result_of::as_map<
typename fusion::result_of::join<
Defs const, Defs2 const>::type>::type> result_type;
return result_type(fusion::as_map(fusion::join(defs(), other.defs())));
}

template <typename Defs2>
friend subrule_group<
typename fusion::result_of::as_map<
typename fusion::result_of::join<
Defs const, Defs2 const>::type>::type>
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
operator,(subrule_group&& left, subrule_group<Defs2>&& other)
#else
operator,(subrule_group& left, subrule_group<Defs2>& other)
#endif
{
return static_cast<subrule_group const&>(left)
.operator,(static_cast<subrule_group<Defs2> const&>(other));
}

parser_type const& get_parameterized_subject() const { return parser(); }
typedef parser_type parameterized_subject_type;
#include <boost/spirit/home/qi/nonterminal/detail/fcall.hpp>
};

template <
int ID_
, typename Locals
, typename Attr
, typename AttrRef
, typename Parameters
, size_t ParamsSize
, typename Subject
, bool Auto_
>
struct subrule_definition
{
typedef mpl::int_<ID_> id_type;
BOOST_STATIC_CONSTANT(int, ID = ID_);

typedef Locals locals_type;
typedef Attr attr_type;
typedef AttrRef attr_reference_type;
typedef Parameters parameter_types;
static size_t const params_size = ParamsSize;

typedef Subject subject_type;
typedef mpl::bool_<Auto_> auto_type;
BOOST_STATIC_CONSTANT(bool, Auto = Auto_);

typedef spirit::qi::detail::parser_binder<
Subject, auto_type> binder_type;

subrule_definition(Subject const& subject, std::string const& name)
: binder(subject), name(name)
{
}

binder_type const binder;
std::string const name;
};

template <
int ID_
, typename T1 = unused_type
, typename T2 = unused_type
>
struct subrule
: proto::extends<
typename proto::terminal<
spirit::qi::reference<subrule<ID_, T1, T2> const>
>::type
, subrule<ID_, T1, T2>
>
, spirit::qi::parser<subrule<ID_, T1, T2> >
{
typedef mpl::int_<ID_> id_type;
BOOST_STATIC_CONSTANT(int, ID = ID_);

typedef subrule<ID_, T1, T2> this_type;
typedef spirit::qi::reference<this_type const> reference_;
typedef typename proto::terminal<reference_>::type terminal;
typedef proto::extends<terminal, this_type> base_type;

typedef mpl::vector<T1, T2> template_params;

typedef typename
spirit::detail::extract_locals<template_params>::type
locals_type;

typedef typename
spirit::detail::extract_encoding<template_params>::type
encoding_type;

typedef typename
spirit::detail::extract_sig<template_params, encoding_type
, spirit::qi::domain>::type
sig_type;

typedef typename
spirit::detail::attr_from_sig<sig_type>::type
attr_type;
BOOST_STATIC_ASSERT_MSG(
!is_reference<attr_type>::value,
"Reference qualifier on Qi subrule attribute type is meaningless");
typedef attr_type& attr_reference_type;

typedef typename
spirit::detail::params_from_sig<sig_type>::type
parameter_types;

static size_t const params_size =
fusion::result_of::size<parameter_types>::type::value;

explicit subrule(std::string const& name_ = "unnamed-subrule")
: base_type(terminal::make(reference_(*this)))
, name_(name_)
{
}

template <typename Expr, bool Auto>
struct def_type_helper
{
BOOST_SPIRIT_ASSERT_MATCH(spirit::qi::domain, Expr);

typedef typename result_of::compile<
spirit::qi::domain, Expr>::type subject_type;

typedef subrule_definition<
ID_
, locals_type
, attr_type
, attr_reference_type
, parameter_types
, params_size
, subject_type
, Auto
> const type;
};

template <typename Expr, bool Auto>
struct group_type_helper
{
typedef typename def_type_helper<Expr, Auto>::type def_type;

typedef typename
#ifndef BOOST_FUSION_HAS_VARIADIC_MAP
fusion::result_of::make_map<id_type, def_type>::type
#else
fusion::result_of::make_map<id_type>::template apply<def_type>::type
#endif
defs_type;

typedef subrule_group<defs_type> type;
};

template <typename Expr>
typename group_type_helper<Expr, false>::type
operator=(Expr const& expr) const
{
typedef group_type_helper<Expr, false> helper;
typedef typename helper::def_type def_type;
typedef typename helper::type result_type;
return result_type(fusion::make_map<id_type>(
def_type(compile<spirit::qi::domain>(expr), name_)));
}

#define BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR(lhs_ref, rhs_ref)        \
template <typename Expr>                                              \
friend typename group_type_helper<Expr, true>::type                   \
operator%=(subrule lhs_ref sr, Expr rhs_ref expr)                     \
{                                                                     \
typedef group_type_helper<Expr, true> helper;                     \
typedef typename helper::def_type def_type;                       \
typedef typename helper::type result_type;                        \
return result_type(fusion::make_map<id_type>(                     \
def_type(compile<spirit::qi::domain>(expr), sr.name_)));      \
}                                                                     \


BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR(const&, const&)
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR(const&, &&)
#else
BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR(const&, &)
#endif
BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR(&, const&)
#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR(&, &&)
#else
BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR(&, &)
#endif

#undef BOOST_SPIRIT_SUBRULE_MODULUS_ASSIGN_OPERATOR

std::string const& name() const
{
return name_;
}

void name(std::string const& str)
{
name_ = str;
}

template <typename Context, typename Iterator>
struct attribute
{
typedef attr_type type;
};

template <typename Iterator, typename Group
, typename Attributes, typename Locals
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, subrule_context<Group, Attributes, Locals>& context
, Skipper const& skipper, Attribute& attr) const
{
return context.group.template parse_subrule_id<ID_>(
first, last, context, skipper, attr);
}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& , Iterator const& 
, Context& 
, Skipper const& , Attribute& ) const
{
BOOST_SPIRIT_ASSERT_FAIL(Iterator
, subrule_used_outside_subrule_group, (id_type));

return false;
}

template <typename Iterator, typename Group
, typename Attributes, typename Locals
, typename Skipper, typename Attribute
, typename Params>
bool parse(Iterator& first, Iterator const& last
, subrule_context<Group, Attributes, Locals>& context
, Skipper const& skipper, Attribute& attr
, Params const& params) const
{
return context.group.template parse_subrule_id<ID_>(
first, last, context, skipper, attr, params);
}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute
, typename Params>
bool parse(Iterator& , Iterator const& 
, Context& 
, Skipper const& , Attribute& 
, Params const& ) const
{
BOOST_SPIRIT_ASSERT_FAIL(Iterator
, subrule_used_outside_subrule_group, (id_type));

return false;
}

template <typename Context>
info what(Context& ) const
{
return info(name_);
}

this_type const& get_parameterized_subject() const { return *this; }
typedef this_type parameterized_subject_type;
#include <boost/spirit/home/qi/nonterminal/detail/fcall.hpp>

std::string name_;
};
}}}}

#if defined(BOOST_MSVC)
# pragma warning(pop)
#endif

#endif
