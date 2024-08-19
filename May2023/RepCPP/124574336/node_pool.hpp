
#ifndef BOOST_INTERPROCESS_DETAIL_NODE_POOL_HPP
#define BOOST_INTERPROCESS_DETAIL_NODE_POOL_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/intrusive/slist.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/allocators/detail/allocator_common.hpp>
#include <boost/container/detail/node_pool_impl.hpp>
#include <cstddef>



namespace boost {
namespace interprocess {
namespace ipcdetail {



template< class SegmentManager, std::size_t NodeSize, std::size_t NodesPerBlock >
class private_node_pool
:  public boost::container::dtl::
private_node_pool_impl<typename SegmentManager::segment_manager_base_type>
{
typedef boost::container::dtl::private_node_pool_impl
<typename SegmentManager::segment_manager_base_type> base_t;
private_node_pool();
private_node_pool(const private_node_pool &);
private_node_pool &operator=(const private_node_pool &);

public:
typedef SegmentManager              segment_manager;
typedef typename base_t::size_type  size_type;

static const size_type nodes_per_block = NodesPerBlock;
static const size_type nodes_per_chunk = NodesPerBlock;

private_node_pool(segment_manager *segment_mngr)
:  base_t(segment_mngr, NodeSize, NodesPerBlock)
{}

segment_manager* get_segment_manager() const
{  return static_cast<segment_manager*>(base_t::get_segment_manager_base()); }
};


template< class SegmentManager
, std::size_t NodeSize
, std::size_t NodesPerBlock
>
class shared_node_pool
:  public ipcdetail::shared_pool_impl
< private_node_pool
<SegmentManager, NodeSize, NodesPerBlock>
>
{
typedef ipcdetail::shared_pool_impl
< private_node_pool
<SegmentManager, NodeSize, NodesPerBlock>
> base_t;
public:
shared_node_pool(SegmentManager *segment_mgnr)
: base_t(segment_mgnr)
{}
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
