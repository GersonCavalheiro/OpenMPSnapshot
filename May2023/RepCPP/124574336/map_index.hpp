
#ifndef BOOST_INTERPROCESS_MAP_INDEX_HPP
#define BOOST_INTERPROCESS_MAP_INDEX_HPP

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
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/private_adaptive_pool.hpp>
#include <boost/intrusive/detail/minimal_pair_header.hpp>         
#include <boost/intrusive/detail/minimal_less_equal_header.hpp>   


namespace boost {
namespace interprocess {
namespace ipcdetail{

template <class MapConfig>
struct map_index_aux
{
typedef typename MapConfig::key_type            key_type;
typedef typename MapConfig::mapped_type         mapped_type;
typedef std::less<key_type>                     key_less;
typedef std::pair<const key_type, mapped_type>  value_type;

typedef private_adaptive_pool
<value_type,
typename MapConfig::
segment_manager_base>                     allocator_type;

typedef boost::interprocess::map
<key_type,  mapped_type,
key_less, allocator_type>                   index_t;
};

}  

template <class MapConfig>
class map_index
: public ipcdetail::map_index_aux<MapConfig>::index_t
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef ipcdetail::map_index_aux<MapConfig>  index_aux;
typedef typename index_aux::index_t       base_type;
typedef typename MapConfig::
segment_manager_base          segment_manager_base;
#endif   

public:
map_index(segment_manager_base *segment_mngr)
: base_type(typename index_aux::key_less(),
segment_mngr){}

void reserve(typename segment_manager_base::size_type)
{    }

void shrink_to_fit()
{  base_type::get_stored_allocator().deallocate_free_blocks(); }
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

template<class MapConfig>
struct is_node_index
<boost::interprocess::map_index<MapConfig> >
{
static const bool value = true;
};
#endif   

}}   

#include <boost/interprocess/detail/config_end.hpp>

#endif   
