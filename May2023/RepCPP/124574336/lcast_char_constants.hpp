
#ifndef BOOST_LEXICAL_CAST_DETAIL_LCAST_CHAR_CONSTANTS_HPP
#define BOOST_LEXICAL_CAST_DETAIL_LCAST_CHAR_CONSTANTS_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

namespace boost 
{
namespace detail 
{
template < typename Char >
struct lcast_char_constants {
BOOST_STATIC_CONSTANT(Char, zero  = static_cast<Char>('0'));
BOOST_STATIC_CONSTANT(Char, minus = static_cast<Char>('-'));
BOOST_STATIC_CONSTANT(Char, plus = static_cast<Char>('+'));
BOOST_STATIC_CONSTANT(Char, lowercase_e = static_cast<Char>('e'));
BOOST_STATIC_CONSTANT(Char, capital_e = static_cast<Char>('E'));
BOOST_STATIC_CONSTANT(Char, c_decimal_separator = static_cast<Char>('.'));
};
}
} 


#endif 

