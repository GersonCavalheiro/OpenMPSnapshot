
#ifndef BOOST_STACKTRACE_DETAIL_TO_DEC_ARRAY_HPP
#define BOOST_STACKTRACE_DETAIL_TO_DEC_ARRAY_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <boost/array.hpp>

namespace boost { namespace stacktrace { namespace detail {

inline boost::array<char, 40> to_dec_array(std::size_t value) BOOST_NOEXCEPT {
boost::array<char, 40> ret;
if (!value) {
ret[0] = '0';
ret[1] = '\0';
return ret;
}

std::size_t digits = 0;
for (std::size_t value_copy = value; value_copy; value_copy /= 10) {
++ digits;
}

for (std::size_t i = 1; i <= digits; ++i) {
ret[digits - i] = static_cast<char>('0' + (value % 10));
value /= 10;
}

ret[digits] = '\0';

return ret;
}


}}} 

#endif 
