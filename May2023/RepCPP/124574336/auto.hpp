
#if !defined(BOOST_SPIRIT_AUTO_FEBRUARY_7_2012_0159PM)
#define BOOST_SPIRIT_AUTO_FEBRUARY_7_2012_0159PM

#include <boost/config.hpp>
#include <boost/typeof/typeof.hpp>

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>


#if defined(BOOST_NO_CXX11_AUTO_DECLARATIONS)

#define BOOST_SPIRIT_AUTO(domain_, name, expr)                                  \
typedef boost::proto::result_of::                                           \
deep_copy<BOOST_TYPEOF(expr)>::type name##_expr_type;                   \
BOOST_SPIRIT_ASSERT_MATCH(                                                  \
boost::spirit::domain_::domain, name##_expr_type);                      \
BOOST_AUTO(name, boost::proto::deep_copy(expr));                            \


#endif

#endif
