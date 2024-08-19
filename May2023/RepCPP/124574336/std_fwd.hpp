
#ifndef BOOST_INTRUSIVE_DETAIL_STD_FWD_HPP
#define BOOST_INTRUSIVE_DETAIL_STD_FWD_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif


#include <boost/move/detail/std_ns_begin.hpp>
BOOST_MOVE_STD_NS_BEG

template<class T>
struct less;

template<class T>
struct equal_to;

struct input_iterator_tag;
struct forward_iterator_tag;
struct bidirectional_iterator_tag;
struct random_access_iterator_tag;

BOOST_MOVE_STD_NS_END
#include <boost/move/detail/std_ns_end.hpp>

#endif 
