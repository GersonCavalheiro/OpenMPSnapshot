#ifndef BOOST_INTERPROCESS_FLAT_MAP_INDEX_HPP
#define BOOST_INTERPROCESS_FLAT_MAP_INDEX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/containers/flat_map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>         
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>   



namespace boost { namespace interprocess {

#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED

template <class MapConfig>
struct flat_map_index_aux
{
typedef typename MapConfig::key_type            key_type;
typedef typename MapConfig::mapped_type         mapped_type;
typedef typename MapConfig::
segment_manager_base                   segment_manager_base;
typedef std::less<key_type>                     key_less;
typedef std::pair<key_type, mapped_type>        value_type;
typedef allocator<value_type
,segment_manager_base>   allocator_type;
typedef flat_map<key_type,  mapped_type,
key_less, allocator_type>      index_t;
};

#endif   

template <class MapConfig>
class flat_map_index
: public flat_map_index_aux<MapConfig>::index_t
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef flat_map_index_aux<MapConfig>  index_aux;
typedef typename index_aux::index_t    base_type;
typedef typename index_aux::
segment_manager_base          segment_manager_base;
#endif   

public:
flat_map_index(segment_manager_base *segment_mngr)
: base_type(typename index_aux::key_less(),
typename index_aux::allocator_type(segment_mngr))
{}

void reserve(typename segment_manager_base::size_type n)
{  base_type::reserve(n);  }

void shrink_to_fit()
{  base_type::shrink_to_fit();   }
};

}}   
#include <boost/interprocess/detail/config_end.hpp>

#endif   
