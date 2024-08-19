
#ifndef BOOST_INTERPROCESS_CACHED_NODE_ALLOCATOR_HPP
#define BOOST_INTERPROCESS_CACHED_NODE_ALLOCATOR_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/allocators/detail/node_pool.hpp>
#include <boost/interprocess/allocators/detail/allocator_common.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/containers/version_type.hpp>
#include <boost/interprocess/allocators/detail/node_tools.hpp>
#include <cstddef>


namespace boost {
namespace interprocess {


#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace ipcdetail {

template < class T
, class SegmentManager
, std::size_t NodesPerBlock = 64
>
class cached_node_allocator_v1
:  public ipcdetail::cached_allocator_impl
< T
, ipcdetail::shared_node_pool
< SegmentManager
, sizeof_value<T>::value
, NodesPerBlock
>
, 1>
{
public:
typedef ipcdetail::cached_allocator_impl
< T
, ipcdetail::shared_node_pool
< SegmentManager
, sizeof_value<T>::value
, NodesPerBlock
>
, 1> base_t;

template<class T2>
struct rebind
{
typedef cached_node_allocator_v1
<T2, SegmentManager, NodesPerBlock>  other;
};

typedef typename base_t::size_type size_type;

cached_node_allocator_v1(SegmentManager *segment_mngr,
size_type max_cached_nodes = base_t::DEFAULT_MAX_CACHED_NODES)
: base_t(segment_mngr, max_cached_nodes)
{}

template<class T2>
cached_node_allocator_v1
(const cached_node_allocator_v1
<T2, SegmentManager, NodesPerBlock> &other)
: base_t(other)
{}
};

}  

#endif   

template < class T
, class SegmentManager
, std::size_t NodesPerBlock
>
class cached_node_allocator
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
:  public ipcdetail::cached_allocator_impl
< T
, ipcdetail::shared_node_pool
< SegmentManager
, sizeof_value<T>::value
, NodesPerBlock
>
, 2>
#endif   
{

#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED
public:
typedef ipcdetail::cached_allocator_impl
< T
, ipcdetail::shared_node_pool
< SegmentManager
, sizeof_value<T>::value
, NodesPerBlock
>
, 2> base_t;

public:
typedef boost::interprocess::version_type<cached_node_allocator, 2>   version;
typedef typename base_t::size_type size_type;

template<class T2>
struct rebind
{
typedef cached_node_allocator<T2, SegmentManager, NodesPerBlock>  other;
};

cached_node_allocator(SegmentManager *segment_mngr,
size_type max_cached_nodes = base_t::DEFAULT_MAX_CACHED_NODES)
: base_t(segment_mngr, max_cached_nodes)
{}

template<class T2>
cached_node_allocator
(const cached_node_allocator<T2, SegmentManager, NodesPerBlock> &other)
: base_t(other)
{}

#else
public:
typedef implementation_defined::segment_manager       segment_manager;
typedef segment_manager::void_pointer                 void_pointer;
typedef implementation_defined::pointer               pointer;
typedef implementation_defined::const_pointer         const_pointer;
typedef T                                             value_type;
typedef typename ipcdetail::add_reference
<value_type>::type                  reference;
typedef typename ipcdetail::add_reference
<const value_type>::type            const_reference;
typedef typename SegmentManager::size_type            size_type;
typedef typename SegmentManager::difference_type      difference_type;

template<class T2>
struct rebind
{
typedef cached_node_allocator<T2, SegmentManager> other;
};

private:
template<class T2, class SegmentManager2, std::size_t N2>
cached_node_allocator& operator=
(const cached_node_allocator<T2, SegmentManager2, N2>&);

cached_node_allocator& operator=(const cached_node_allocator&);

public:
cached_node_allocator(segment_manager *segment_mngr);

cached_node_allocator(const cached_node_allocator &other);

template<class T2>
cached_node_allocator
(const cached_node_allocator<T2, SegmentManager, NodesPerBlock> &other);

~cached_node_allocator();

node_pool_t* get_node_pool() const;

segment_manager* get_segment_manager()const;

size_type max_size() const;

pointer allocate(size_type count, cvoid_pointer hint = 0);

void deallocate(const pointer &ptr, size_type count);

void deallocate_free_blocks();

friend void swap(self_t &alloc1, self_t &alloc2);

pointer address(reference value) const;

const_pointer address(const_reference value) const;

void construct(const pointer &ptr, const_reference v);

void destroy(const pointer &ptr);

size_type size(const pointer &p) const;

pointer allocation_command(boost::interprocess::allocation_type command,
size_type limit_size, size_type &prefer_in_recvd_out_size, pointer &reuse);

void allocate_many(size_type elem_size, size_type num_elements, multiallocation_chain &chain);

void allocate_many(const size_type *elem_sizes, size_type n_elements, multiallocation_chain &chain);

void deallocate_many(multiallocation_chain &chain);

pointer allocate_one();

multiallocation_chain allocate_individual(size_type num_elements);

void deallocate_one(const pointer &p);

void deallocate_individual(multiallocation_chain it);
void set_max_cached_nodes(size_type newmax);

size_type get_max_cached_nodes() const;
#endif
};

#ifdef BOOST_INTERPROCESS_DOXYGEN_INVOKED

template<class T, class S, std::size_t NPC> inline
bool operator==(const cached_node_allocator<T, S, NPC> &alloc1,
const cached_node_allocator<T, S, NPC> &alloc2);

template<class T, class S, std::size_t NPC> inline
bool operator!=(const cached_node_allocator<T, S, NPC> &alloc1,
const cached_node_allocator<T, S, NPC> &alloc2);

#endif

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

