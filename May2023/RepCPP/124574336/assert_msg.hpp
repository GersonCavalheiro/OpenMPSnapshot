
#if !defined(BOOST_SPIRIT_ASSERT_MSG_JUN_23_2009_0836AM)
#define BOOST_SPIRIT_ASSERT_MSG_JUN_23_2009_0836AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#if !defined(BOOST_SPIRIT_DONT_USE_MPL_ASSERT_MSG)
# if defined(BOOST_MSVC) && BOOST_MSVC < 1500
#  define BOOST_SPIRIT_DONT_USE_MPL_ASSERT_MSG 1
# endif
#endif

#if !defined(BOOST_NO_CXX11_STATIC_ASSERT) || BOOST_SPIRIT_DONT_USE_MPL_ASSERT_MSG != 0
#include <boost/static_assert.hpp>
#define BOOST_SPIRIT_ASSERT_MSG(Cond, Msg, Types)                             \
BOOST_STATIC_ASSERT_MSG(Cond, # Msg)
#else
#include <boost/mpl/assert.hpp>
#define BOOST_SPIRIT_ASSERT_MSG(Cond, Msg, Types)                             \
BOOST_MPL_ASSERT_MSG(Cond, Msg, Types)
#endif

#define BOOST_SPIRIT_ASSERT_MATCH(Domain, Expr)                               \
BOOST_SPIRIT_ASSERT_MSG((                                             \
boost::spirit::traits::matches< Domain, Expr >::value             \
), error_invalid_expression, (Expr))

#include <boost/type_traits/is_same.hpp>

#define BOOST_SPIRIT_ASSERT_FAIL(TemplateParam, Msg, Types)                   \
BOOST_SPIRIT_ASSERT_MSG((!boost::is_same<                             \
TemplateParam, TemplateParam >::value), Msg, Types)

#endif

