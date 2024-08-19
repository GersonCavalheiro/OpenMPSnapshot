
#if !defined(BOOST_SPIRIT_COPY_FEBRUARY_7_2012_0159PM)
#define BOOST_SPIRIT_COPY_FEBRUARY_7_2012_0159PM

#include <boost/config.hpp>

#if defined(_MSC_VER)
#pragma once
#endif

#if !defined(BOOST_NO_CXX11_AUTO_DECLARATIONS)
#include <boost/proto/deep_copy.hpp>

namespace boost { namespace spirit { namespace qi
{
template <typename Expr>
typename boost::proto::result_of::deep_copy<Expr>::type
copy(Expr const& expr)
{
BOOST_SPIRIT_ASSERT_MATCH(boost::spirit::qi::domain, Expr);
return boost::proto::deep_copy(expr);
}
}}}

#endif
#endif
