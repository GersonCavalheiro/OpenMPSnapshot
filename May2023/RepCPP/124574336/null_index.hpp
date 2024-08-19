#ifndef BOOST_INTERPROCESS_NULL_INDEX_HPP
#define BOOST_INTERPROCESS_NULL_INDEX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/offset_ptr.hpp>


namespace boost {
namespace interprocess {

template <class MapConfig>
class null_index
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef typename MapConfig::
segment_manager_base    segment_manager_base;
#endif   

public:
typedef int * iterator;
typedef const int * const_iterator;

const_iterator begin() const
{  return const_iterator(0);  }

iterator begin()
{  return iterator(0);  }

const_iterator end() const
{  return const_iterator(0);  }

iterator end()
{  return iterator(0);  }

null_index(segment_manager_base *){}
};

}}   

#include <boost/interprocess/detail/config_end.hpp>

#endif   
