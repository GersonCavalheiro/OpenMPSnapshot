
#if !defined(BOOST_SPIRIT_KARMA_PHOENIX_ATTRIBUTES_OCT_01_2009_1128AM)
#define BOOST_SPIRIT_KARMA_PHOENIX_ATTRIBUTES_OCT_01_2009_1128AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/include/version.hpp>

#if SPIRIT_VERSION >= 0x2020

#include <boost/spirit/home/karma/detail/attributes.hpp>
#include <boost/spirit/home/karma/detail/indirect_iterator.hpp>
#include <boost/spirit/home/support/container.hpp>

#include <boost/utility/result_of.hpp>

namespace boost { namespace phoenix
{
template <typename Expr>
struct actor;
}}

namespace boost { namespace spirit { namespace traits
{
template <typename Eval>
struct is_container<phoenix::actor<Eval> const>
: is_container<typename boost::result_of<phoenix::actor<Eval>()>::type>
{};

template <typename Eval>
struct container_iterator<phoenix::actor<Eval> const>
{
typedef phoenix::actor<Eval> const& type;
};

template <typename Eval>
struct begin_container<phoenix::actor<Eval> const>
{
typedef phoenix::actor<Eval> const& type;
static type call(phoenix::actor<Eval> const& f)
{
return f;
}
};

template <typename Eval>
struct end_container<phoenix::actor<Eval> const>
{
typedef phoenix::actor<Eval> const& type;
static type call(phoenix::actor<Eval> const& f)
{
return f;
}
};

template <typename Eval>
struct deref_iterator<phoenix::actor<Eval> const>
{
typedef typename boost::result_of<phoenix::actor<Eval>()>::type type;
static type call(phoenix::actor<Eval> const& f)
{
return f();
}
};

template <typename Eval>
struct next_iterator<phoenix::actor<Eval> const>
{
typedef phoenix::actor<Eval> const& type;
static type call(phoenix::actor<Eval> const& f)
{
return f;
}
};

template <typename Eval>
struct compare_iterators<phoenix::actor<Eval> const>
{
static bool 
call(phoenix::actor<Eval> const&, phoenix::actor<Eval> const&)
{
return false;
}
};

template <typename Eval>
struct container_value<phoenix::actor<Eval> >
{
typedef phoenix::actor<Eval> const& type;
};

template <typename Eval>
struct make_indirect_iterator<phoenix::actor<Eval> const>
{
typedef phoenix::actor<Eval> const& type;
};

template <typename Eval, typename Exposed>
struct extract_from_attribute<phoenix::actor<Eval>, Exposed>
{
typedef typename boost::result_of<phoenix::actor<Eval>()>::type type;

template <typename Context>
static type call(phoenix::actor<Eval> const& f, Context& context)
{
return f(unused, context);
}
};
}}}

#endif
#endif
