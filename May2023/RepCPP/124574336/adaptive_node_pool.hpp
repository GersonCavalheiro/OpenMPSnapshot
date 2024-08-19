
#ifndef BOOST_INTERPROCESS_DETAIL_ADAPTIVE_NODE_POOL_HPP
#define BOOST_INTERPROCESS_DETAIL_ADAPTIVE_NODE_POOL_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/detail/math_functions.hpp>
#include <boost/intrusive/set.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/interprocess/detail/type_traits.hpp>
#include <boost/interprocess/mem_algo/detail/mem_algo_common.hpp>
#include <boost/interprocess/allocators/detail/node_tools.hpp>
#include <boost/interprocess/allocators/detail/allocator_common.hpp>
#include <cstddef>
#include <boost/config/no_tr1/cmath.hpp>
#include <boost/container/detail/adaptive_node_pool_impl.hpp>
#include <boost/assert.hpp>


namespace boost {
namespace interprocess {
namespace ipcdetail {

template< class SegmentManager
, std::size_t NodeSize
, std::size_t NodesPerBlock
, std::size_t MaxFreeBlocks
, unsigned char OverheadPercent
>
class private_adaptive_node_pool
:  public boost::container::dtl::private_adaptive_node_pool_impl_rt
< typename SegmentManager::segment_manager_base_type
, ::boost::container::adaptive_pool_flag::size_ordered |
::boost::container::adaptive_pool_flag::address_ordered
>
{
typedef boost::container::dtl::private_adaptive_node_pool_impl_rt
< typename SegmentManager::segment_manager_base_type
, ::boost::container::adaptive_pool_flag::size_ordered |
::boost::container::adaptive_pool_flag::address_ordered
> base_t;
private_adaptive_node_pool();
private_adaptive_node_pool(const private_adaptive_node_pool &);
private_adaptive_node_pool &operator=(const private_adaptive_node_pool &);

public:
typedef SegmentManager              segment_manager;
typedef typename base_t::size_type  size_type;

static const size_type nodes_per_block = NodesPerBlock;

private_adaptive_node_pool(segment_manager *segment_mngr)
:  base_t(segment_mngr, NodeSize, NodesPerBlock, MaxFreeBlocks, OverheadPercent)
{}

segment_manager* get_segment_manager() const
{  return static_cast<segment_manager*>(base_t::get_segment_manager_base()); }
};

template< class SegmentManager
, std::size_t NodeSize
, std::size_t NodesPerBlock
, std::size_t MaxFreeBlocks
, unsigned char OverheadPercent
>
class shared_adaptive_node_pool
:  public ipcdetail::shared_pool_impl
< private_adaptive_node_pool
<SegmentManager, NodeSize, NodesPerBlock, MaxFreeBlocks, OverheadPercent>
>
{
typedef ipcdetail::shared_pool_impl
< private_adaptive_node_pool
<SegmentManager, NodeSize, NodesPerBlock, MaxFreeBlocks, OverheadPercent>
> base_t;
public:
shared_adaptive_node_pool(SegmentManager *segment_mgnr)
: base_t(segment_mgnr)
{}
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
