#ifndef BOOST_MOVE_DETAIL_PLACEMENT_NEW_HPP
#define BOOST_MOVE_DETAIL_PLACEMENT_NEW_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

struct boost_move_new_t{};

inline void *operator new(std::size_t, void *p, boost_move_new_t)
{  return p;  }

inline void operator delete(void *, void *, boost_move_new_t)
{}

#endif   
