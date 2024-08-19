
#if !defined(BOOST_SPIRIT_GENERATOR_BINDER_APR_17_2009_0952PM)
#define BOOST_SPIRIT_GENERATOR_BINDER_APR_17_2009_0952PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/fusion/include/at.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>

namespace boost { namespace spirit { namespace karma { namespace detail
{
template <typename Generator, typename Auto>
struct generator_binder
{
generator_binder(Generator const& g)
: g(g) {}

template <typename OutputIterator, typename Delimiter, typename Context>
bool call(OutputIterator& sink, Context& context
, Delimiter const& delim, mpl::true_) const
{
return g.generate(sink, context, delim, unused);
}

template <typename OutputIterator, typename Delimiter, typename Context>
bool call(OutputIterator& sink, Context& context
, Delimiter const& delim, mpl::false_) const
{
return g.generate(sink, context, delim
, fusion::at_c<0>(context.attributes));
}

template <typename OutputIterator, typename Delimiter, typename Context>
bool operator()(OutputIterator& sink, Context& context
, Delimiter const& delim) const
{
typedef typename traits::has_semantic_action<Generator>::type auto_rule;
return call(sink, context, delim, auto_rule());
}

Generator g;
};

template <typename Generator>
struct generator_binder<Generator, mpl::true_>
{
generator_binder(Generator const& g)
: g(g) {}

template <typename OutputIterator, typename Delimiter, typename Context>
bool operator()(OutputIterator& sink, Context& context
, Delimiter const& delim) const
{
return g.generate(sink, context, delim
, fusion::at_c<0>(context.attributes));
}

Generator g;
};

template <typename Auto, typename Generator>
inline generator_binder<Generator, Auto>
bind_generator(Generator const& g)
{
return generator_binder<Generator, Auto>(g);
}

}}}}

#endif
