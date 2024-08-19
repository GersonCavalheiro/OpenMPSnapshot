
#ifndef BOOST_LEXICAL_CAST_DETAIL_WIDEST_CHAR_HPP
#define BOOST_LEXICAL_CAST_DETAIL_WIDEST_CHAR_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif


#include <boost/type_traits/conditional.hpp>

namespace boost { namespace detail {

template <typename TargetChar, typename SourceChar>
struct widest_char {
typedef BOOST_DEDUCED_TYPENAME boost::conditional<
(sizeof(TargetChar) > sizeof(SourceChar))
, TargetChar
, SourceChar
>::type type;
};

}} 

#endif 

