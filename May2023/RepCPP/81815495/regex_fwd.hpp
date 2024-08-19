
#ifndef ASIO_DETAIL_REGEX_FWD_HPP
#define ASIO_DETAIL_REGEX_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#if defined(ASIO_HAS_BOOST_REGEX)

#include <boost/regex_fwd.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION >= 107600
# if defined(BOOST_REGEX_CXX03)
#  include <boost/regex/v4/match_flags.hpp>
# else 
#  include <boost/regex/v5/match_flags.hpp>
# endif 
#else 
# include <boost/regex/v4/match_flags.hpp>
#endif 

namespace boost {

template <class BidiIterator>
struct sub_match;

template <class BidiIterator, class Allocator>
class match_results;

} 

#endif 

#endif 
