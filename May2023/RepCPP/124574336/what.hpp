
#if !defined(BOOST_SPIRIT_WHAT_APRIL_21_2007_0732AM)
#define BOOST_SPIRIT_WHAT_APRIL_21_2007_0732AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/assert.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>

namespace boost { namespace spirit { namespace qi
{
template <typename Expr>
inline info what(Expr const& expr)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Expr);
return compile<qi::domain>(expr).what(unused);
}
}}}

#endif

