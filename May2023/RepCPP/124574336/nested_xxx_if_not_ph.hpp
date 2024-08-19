

#ifndef BOOST_FLYWEIGHT_DETAIL_NESTED_XXX_IF_NOT_PH_HPP
#define BOOST_FLYWEIGHT_DETAIL_NESTED_XXX_IF_NOT_PH_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/flyweight/detail/is_placeholder_expr.hpp>
#include <boost/mpl/if.hpp>



#define BOOST_FLYWEIGHT_NESTED_XXX_IF_NOT_PLACEHOLDER_EXPRESSION_DEF(name) \
struct nested_##name##_if_not_placeholder_expression_helper                \
{                                                                          \
typedef int name;                                                        \
};                                                                         \
\
template<typename T>                                                       \
struct nested_##name##_if_not_placeholder_expression                       \
{                                                                          \
typedef typename boost::mpl::if_<                                        \
boost::flyweights::detail::is_placeholder_expression<T>,               \
nested_##name##_if_not_placeholder_expression_helper,                  \
T                                                                      \
>::type::name type;                                                      \
};

#endif
