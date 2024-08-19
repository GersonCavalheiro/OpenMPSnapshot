
#ifndef BOOST_STACKTRACE_DETAIL_TRY_DEC_CONVERT_HPP
#define BOOST_STACKTRACE_DETAIL_TRY_DEC_CONVERT_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <cstdlib>

namespace boost { namespace stacktrace { namespace detail {

inline bool try_dec_convert(const char* s, std::size_t& res) BOOST_NOEXCEPT {
char* end_ptr = 0;
res = std::strtoul(s, &end_ptr, 10);
return *end_ptr == '\0';
}


}}} 

#endif 
