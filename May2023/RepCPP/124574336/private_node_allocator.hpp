
#ifndef BOOST_INTERPROCESS_PRIVATE_NODE_ALLOCATOR_HPP
#define BOOST_INTERPROCESS_PRIVATE_NODE_ALLOCATOR_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/intrusive/pointer_traits.hpp>

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/assert.hpp>
#include <boost/utility/addressof.hpp>
#include <boost/interprocess/allocators/detail/node_pool.hpp>
#include <boost/interprocess/containers/version_type.hpp>
#include <boost/container/detail/multiallocation_chain.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/move/adl_move_swap.hpp>
#include <cstddef>


namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

namespace ipcdetail {

template < unsigned int Version
, class T
, class SegmentManager
, std::size_t NodesPerBlock
>
class private_node_allocator_base
: public node_pool_allocation_impl
< private_node_allocator_base < Version, T, SegmentManager, NodesPerBlock>
, Version
, T
, SegmentManager
>
{
public:
typedef SegmentManager                                segment_manager;
typedef typename SegmentManager::void_pointer         void_pointer;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef private_node_allocator_base
< Version, T, SegmentManager, NodesPerBlock>       self_t;
typedef ipcdetail::private_node_pool
<SegmentManager
, sizeof_value<T>::value
, NodesPerBlock
> node_pool_t;

BOOST_STATIC_ASSERT((Version <=2));

#endif   

public:

typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<T>::type                         pointer;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<const T>::type                   const_pointer;
typedef T                                             value_type;
typedef typename ipcdetail::add_reference
<value_type>::type                  reference;
typedef typename ipcdetail::add_reference
<const value_type>::type            const_reference;
typedef typename segment_manager::size_type           size_type;
typedef typename segment_manager::difference_type     difference_type;
typedef boost::interprocess::version_type
<private_node_allocator_base, Version>              version;
typedef boost::container::dtl::transform_multiallocation_chain
<typename SegmentManager::multiallocation_chain, T>multiallocation_chain;

template<class T2>
struct rebind
{
typedef private_node_allocator_base
<Version, T2, SegmentManager, NodesPerBlock>   other;
};

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
template <int dummy>
struct node_pool
{
typedef ipcdetail::private_node_pool
<SegmentManager
, sizeof_value<T>::value
, NodesPerBlock
> type;

static type *get(void *p)
{  return static_cast<type*>(p);  }
};

private:
template<unsigned int Version2, class T2, class MemoryAlgorithm2, std::size_t N2>
private_node_allocator_base& operator=
(const private_node_allocator_base<Version2, T2, MemoryAlgorithm2, N2>&);

private_node_allocator_base& operator=(const private_node_allocator_base&);
#endif   

public:
private_node_allocator_base(segment_manager *segment_mngr)
: m_node_pool(segment_mngr)
{}

private_node_allocator_base(const private_node_allocator_base &other)
: m_node_pool(other.get_segment_manager())
{}

template<class T2>
private_node_allocator_base
(const private_node_allocator_base
<Version, T2, SegmentManager, NodesPerBlock> &other)
: m_node_pool(other.get_segment_manager())
{}

~private_node_allocator_base()
{}

segment_manager* get_segment_manager()const
{  return m_node_pool.get_segment_manager(); }

node_pool_t* get_node_pool() const
{  return const_cast<node_pool_t*>(&m_node_pool); }

friend void swap(self_t &alloc1,self_t &alloc2)
{  boost::adl_move_swap(alloc1.m_node_pool, alloc2.m_node_pool);  }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
node_pool_t m_node_pool;
#endif   
};

template<unsigned int V, class T, class S, std::size_t NPC> inline
bool operator==(const private_node_allocator_base<V, T, S, NPC> &alloc1,
const private_node_allocator_base<V, T, S, NPC> &alloc2)
{  return &alloc1 == &alloc2; }

template<unsigned int V, class T, class S, std::size_t NPC> inline
bool operator!=(const private_node_allocator_base<V, T, S, NPC> &alloc1,
const private_node_allocator_base<V, T, S, NPC> &alloc2)
{  return &alloc1 != &alloc2; }

template < class T
, class SegmentManager
, std::size_t NodesPerBlock = 64
>
class private_node_allocator_v1
:  public private_node_allocator_base
< 1
, T
, SegmentManager
, NodesPerBlock
>
{
public:
typedef ipcdetail::private_node_allocator_base
< 1, T, SegmentManager, NodesPerBlock> base_t;

template<class T2>
struct rebind
{
typedef private_node_allocator_v1<T2, SegmentManager, NodesPerBlock>  other;
};

private_node_allocator_v1(SegmentManager *segment_mngr)
: base_t(segment_mngr)
{}

template<class T2>
private_node_allocator_v1
(const private_node_allocator_v1<T2, SegmentManager, NodesPerBlock> &other)
: base_t(other)
{}
};

}  

#endif   

template < class T
, class SegmentManager
, std::size_t NodesPerBlock
>
class private_node_allocator
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
:  public ipcdetail::private_node_allocator_base
< 2
, T
, SegmentManager
, NodesPerBlock
>
#endif   
{

#ifndef BOOST_INTERPROCESS_DOXYGEN_INVOKED
typedef ipcdetail::private_node_allocator_base
< 2, T, SegmentManager, NodesPerBlock> base_t;
public:
typedef boost::interprocess::version_type<private_node_allocator, 2>   version;

template<class T2>
struct rebind
{
typedef private_node_allocator
<T2, SegmentManager, NodesPerBlock>  other;
};

private_node_allocator(SegmentManager *segment_mngr)
: base_t(segment_mngr)
{}

template<class T2>
private_node_allocator
(const private_node_allocator<T2, SegmentManager, NodesPerBlock> &other)
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
typedef typename segment_manager::size_type           size_type;
typedef typename segment_manage::difference_type      difference_type;

template<class T2>
struct rebind
{
typedef private_node_allocator
<T2, SegmentManager, NodesPerBlock> other;
};

private:
template<class T2, class SegmentManager2, std::size_t N2>
private_node_allocator& operator=
(const private_node_allocator<T2, SegmentManager2, N2>&);

private_node_allocator& operator=(const private_node_allocator&);

public:
private_node_allocator(segment_manager *segment_mngr);

private_node_allocator(const private_node_allocator &other);

template<class T2>
private_node_allocator
(const private_node_allocator<T2, SegmentManager, NodesPerBlock> &other);

~private_node_allocator();

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

void allocate_individual(size_type num_elements, multiallocation_chain &chain);

void deallocate_one(const pointer &p);

void deallocate_individual(multiallocation_chain &chain);
#endif
};

#ifdef BOOST_INTERPROCESS_DOXYGEN_INVOKED

template<class T, class S, std::size_t NodesPerBlock, std::size_t F, unsigned char OP> inline
bool operator==(const private_node_allocator<T, S, NodesPerBlock, F, OP> &alloc1,
const private_node_allocator<T, S, NodesPerBlock, F, OP> &alloc2);

template<class T, class S, std::size_t NodesPerBlock, std::size_t F, unsigned char OP> inline
bool operator!=(const private_node_allocator<T, S, NodesPerBlock, F, OP> &alloc1,
const private_node_allocator<T, S, NodesPerBlock, F, OP> &alloc2);

#endif

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

