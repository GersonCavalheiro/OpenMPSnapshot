
#ifndef BOOST_TT_HAS_LOGICAL_NOT_HPP_INCLUDED
#define BOOST_TT_HAS_LOGICAL_NOT_HPP_INCLUDED

#if defined(__GNUC__) && (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__ > 40800)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif

#define BOOST_TT_TRAIT_NAME has_logical_not
#define BOOST_TT_TRAIT_OP !
#define BOOST_TT_FORBIDDEN_IF\
false

#include <boost/type_traits/detail/has_prefix_operator.hpp>

#undef BOOST_TT_TRAIT_NAME
#undef BOOST_TT_TRAIT_OP
#undef BOOST_TT_FORBIDDEN_IF

#if defined(__GNUC__) && (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__ > 40800)
#pragma GCC diagnostic pop
#endif

#endif
