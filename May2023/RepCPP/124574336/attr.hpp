
#if !defined(BOOST_SPIRIT_ATTR_JUL_23_2008_0956AM)
#define BOOST_SPIRIT_ATTR_JUL_23_2008_0956AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/spirit/home/qi/domain.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_const.hpp>

namespace boost { namespace spirit
{
template <typename A0>       
struct use_terminal<
qi::domain, terminal_ex<tag::attr, fusion::vector1<A0> > >
: mpl::true_ {};

template <>                  
struct use_lazy_terminal<qi::domain, tag::attr, 1>
: mpl::true_ {};

}}

namespace boost { namespace spirit { namespace qi
{
#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
using spirit::attr;
#endif
using spirit::attr_type;

template <typename Value>
struct attr_parser : primitive_parser<attr_parser<Value> >
{
template <typename Context, typename Iterator>
struct attribute : remove_const<Value> {};

attr_parser(typename add_reference<Value>::type value)
: value_(value) {}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& , Iterator const& 
, Context& , Skipper const& 
, Attribute& attr_) const
{
spirit::traits::assign_to(value_, attr_);
return true;        
}

template <typename Context>
info what(Context& ) const
{
return info("attr");
}

Value value_;

BOOST_DELETED_FUNCTION(attr_parser& operator= (attr_parser const&))
};

template <typename Modifiers, typename A0>
struct make_primitive<
terminal_ex<tag::attr, fusion::vector1<A0> >
, Modifiers>
{
typedef typename add_const<A0>::type const_value;
typedef attr_parser<const_value> result_type;

template <typename Terminal>
result_type operator()(Terminal const& term, unused_type) const
{
return result_type(fusion::at_c<0>(term.args));
}
};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename T, typename Attr, typename Context, typename Iterator>
struct handles_container<qi::attr_parser<T>, Attr, Context, Iterator>
: traits::is_container<Attr> {}; 
}}}

#endif


