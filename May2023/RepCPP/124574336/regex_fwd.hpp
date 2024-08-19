
#ifndef BOOST_ASIO_DETAIL_REGEX_FWD_HPP
#define BOOST_ASIO_DETAIL_REGEX_FWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#if defined(BOOST_ASIO_HAS_BOOST_REGEX)

#include <boost/regex_fwd.hpp>
#include <boost/regex/v4/match_flags.hpp>

namespace boost {

template <class BidiIterator>
struct sub_match;

template <class BidiIterator, class Allocator>
class match_results;

} 

#endif 

#endif 
