
#ifndef BOOST_INTERPROCESS_UNORDERED_MAP_INDEX_HPP
#define BOOST_INTERPROCESS_UNORDERED_MAP_INDEX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/intrusive/detail/minimal_pair_header.hpp>
#include <boost/unordered_map.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/allocators/private_adaptive_pool.hpp>

#include <boost/intrusive/detail/minimal_pair_header.hpp>         
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>   


namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template <class MapConfig>
struct unordered_map_index_aux
{
typedef typename MapConfig::key_type            key_type;
typedef typename MapConfig::mapped_type         mapped_type;
typedef std::equal_to<key_type>                 key_equal;
typedef std::pair<const key_type, mapped_type>  value_type;
typedef private_adaptive_pool
<value_type,
typename MapConfig::
segment_manager_base>      allocator_type;
struct hasher
{
typedef key_type argument_type;
typedef std::size_t result_type;

std::size_t operator()(const key_type &val) const
{
typedef typename key_type::char_type    char_type;
const char_type *beg = ipcdetail::to_raw_pointer(val.mp_str),
*end = beg + val.m_len;
return boost::hash_range(beg, end);
}
};
typedef unordered_map<key_type,  mapped_type, hasher,
key_equal, allocator_type>      index_t;
};

#endif   

template <class MapConfig>
class unordered_map_index
: public unordered_map_index_aux<MapConfig>::index_t
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef unordered_map_index_aux<MapConfig>   index_aux;
typedef typename index_aux::index_t          base_type;
typedef typename
MapConfig::segment_manager_base     segment_manager_base;
#endif   

public:
unordered_map_index(segment_manager_base *segment_mngr)
: base_type(0,
typename index_aux::hasher(),
typename index_aux::key_equal(),
segment_mngr){}

void reserve(typename segment_manager_base::size_type n)
{  base_type::rehash(n);  }

void shrink_to_fit()
{  base_type::rehash(base_type::size()); }
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template<class MapConfig>
struct is_node_index
<boost::interprocess::unordered_map_index<MapConfig> >
{
static const bool value = true;
};
#endif   

}}   

#include <boost/interprocess/detail/config_end.hpp>

#endif   
