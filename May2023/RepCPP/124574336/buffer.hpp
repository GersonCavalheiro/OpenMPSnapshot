
#ifndef BOOST_SPIRIT_KARMA_DIRECTIVE_BUFFER_HPP
#define BOOST_SPIRIT_KARMA_DIRECTIVE_BUFFER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/karma/meta_compiler.hpp>
#include <boost/spirit/home/karma/generator.hpp>
#include <boost/spirit/home/karma/domain.hpp>
#include <boost/spirit/home/karma/detail/output_iterator.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/karma/detail/attributes.hpp>

namespace boost { namespace spirit
{
template <>
struct use_directive<karma::domain, tag::buffer> 
: mpl::true_ {};

}}

namespace boost { namespace spirit { namespace karma
{
#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
using spirit::buffer;
#endif
using spirit::buffer_type;

template <typename Subject>
struct buffer_directive : unary_generator<buffer_directive<Subject> >
{
typedef Subject subject_type;
typedef mpl::int_<
subject_type::properties::value | 
generator_properties::countingbuffer
> properties;

buffer_directive(Subject const& subject)
: subject(subject) {}

template <typename Context, typename Iterator>
struct attribute
: traits::attribute_of<subject_type, Context, Iterator>
{};

template <typename OutputIterator, typename Context, typename Delimiter
, typename Attribute>
bool generate(OutputIterator& sink, Context& ctx, Delimiter const& d
, Attribute const& attr) const
{
detail::enable_buffering<OutputIterator> buffering(sink);
bool r = false;
{
detail::disable_counting<OutputIterator> nocounting(sink);
r = subject.generate(sink, ctx, d, attr);
}
if (r) 
buffering.buffer_copy();
return r;
}

template <typename Context>
info what(Context& context) const
{
return info("buffer", subject.what(context));
}

Subject subject;
};

template <typename Subject, typename Modifiers>
struct make_directive<tag::buffer, Subject, Modifiers>
{
typedef buffer_directive<Subject> result_type;
result_type operator()(unused_type, Subject const& subject
, unused_type) const
{
return result_type(subject);
}
};

template <typename Subject, typename Modifiers>
struct make_directive<tag::buffer, buffer_directive<Subject>, Modifiers>
{
typedef buffer_directive<Subject> result_type;
result_type operator()(unused_type
, buffer_directive<Subject> const& subject, unused_type) const
{
return subject;
}
};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Subject>
struct has_semantic_action<karma::buffer_directive<Subject> >
: unary_has_semantic_action<Subject> {};

template <typename Subject, typename Attribute, typename Context
, typename Iterator>
struct handles_container<karma::buffer_directive<Subject>, Attribute
, Context, Iterator>
: unary_handles_container<Subject, Attribute, Context, Iterator> {};
}}}

#endif
