
#ifndef BOOST_INTERPROCESS_ALLOCATOR_DETAIL_ALLOCATOR_COMMON_HPP
#define BOOST_INTERPROCESS_ALLOCATOR_DETAIL_ALLOCATOR_COMMON_HPP

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
#include <boost/interprocess/detail/utilities.hpp> 
#include <boost/utility/addressof.hpp> 
#include <boost/assert.hpp>   
#include <boost/interprocess/exceptions.hpp> 
#include <boost/interprocess/sync/scoped_lock.hpp> 
#include <boost/interprocess/containers/allocation_type.hpp> 
#include <boost/container/detail/multiallocation_chain.hpp>
#include <boost/interprocess/mem_algo/detail/mem_algo_common.hpp>
#include <boost/interprocess/detail/segment_manager_helper.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/interprocess/detail/type_traits.hpp>
#include <boost/interprocess/detail/utilities.hpp>
#include <boost/container/detail/placement_new.hpp>
#include <boost/move/adl_move_swap.hpp>

namespace boost {
namespace interprocess {

template <class T>
struct sizeof_value
{
static const std::size_t value = sizeof(T);
};

template <>
struct sizeof_value<void>
{
static const std::size_t value = sizeof(void*);
};

template <>
struct sizeof_value<const void>
{
static const std::size_t value = sizeof(void*);
};

template <>
struct sizeof_value<volatile void>
{
static const std::size_t value = sizeof(void*);
};

template <>
struct sizeof_value<const volatile void>
{
static const std::size_t value = sizeof(void*);
};

namespace ipcdetail {

template<class NodePool>
struct get_or_create_node_pool_func
{

void operator()()
{
mp_node_pool =    mp_segment_manager->template find_or_construct
<NodePool>(boost::interprocess::unique_instance)(mp_segment_manager);
if(mp_node_pool != 0)
mp_node_pool->inc_ref_count();
}

get_or_create_node_pool_func(typename NodePool::segment_manager *mngr)
: mp_segment_manager(mngr){}

NodePool                            *mp_node_pool;
typename NodePool::segment_manager  *mp_segment_manager;
};

template<class NodePool>
inline NodePool *get_or_create_node_pool(typename NodePool::segment_manager *mgnr)
{
ipcdetail::get_or_create_node_pool_func<NodePool> func(mgnr);
mgnr->atomic_func(func);
return func.mp_node_pool;
}

template<class NodePool>
struct destroy_if_last_link_func
{
void operator()()
{
if(mp_node_pool->dec_ref_count() != 0) return;

mp_node_pool->get_segment_manager()->template destroy<NodePool>(boost::interprocess::unique_instance);
}

destroy_if_last_link_func(NodePool *pool)
: mp_node_pool(pool)
{}

NodePool                           *mp_node_pool;
};

template<class NodePool>
inline void destroy_node_pool_if_last_link(NodePool *pool)
{
typename NodePool::segment_manager *mngr = pool->get_segment_manager();
destroy_if_last_link_func<NodePool>func(pool);
mngr->atomic_func(func);
}

template<class NodePool>
class cache_impl
{
typedef typename NodePool::segment_manager::
void_pointer                                          void_pointer;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<NodePool>::type                     node_pool_ptr;
typedef typename NodePool::multiallocation_chain         multiallocation_chain;
typedef typename NodePool::segment_manager::size_type    size_type;
node_pool_ptr                 mp_node_pool;
multiallocation_chain         m_cached_nodes;
size_type                     m_max_cached_nodes;

public:
typedef typename NodePool::segment_manager            segment_manager;

cache_impl(segment_manager *segment_mngr, size_type max_cached_nodes)
: mp_node_pool(get_or_create_node_pool<NodePool>(segment_mngr))
, m_max_cached_nodes(max_cached_nodes)
{}

cache_impl(const cache_impl &other)
: mp_node_pool(other.get_node_pool())
, m_max_cached_nodes(other.get_max_cached_nodes())
{
mp_node_pool->inc_ref_count();
}

~cache_impl()
{
this->deallocate_all_cached_nodes();
ipcdetail::destroy_node_pool_if_last_link(ipcdetail::to_raw_pointer(mp_node_pool));
}

NodePool *get_node_pool() const
{  return ipcdetail::to_raw_pointer(mp_node_pool); }

segment_manager *get_segment_manager() const
{  return mp_node_pool->get_segment_manager(); }

size_type get_max_cached_nodes() const
{  return m_max_cached_nodes; }

void *cached_allocation()
{
if(m_cached_nodes.empty()){
mp_node_pool->allocate_nodes(m_max_cached_nodes/2, m_cached_nodes);
}
void *ret = ipcdetail::to_raw_pointer(m_cached_nodes.pop_front());
return ret;
}

void cached_allocation(size_type n, multiallocation_chain &chain)
{
size_type count = n, allocated(0);
BOOST_TRY{
while(!m_cached_nodes.empty() && count--){
void *ret = ipcdetail::to_raw_pointer(m_cached_nodes.pop_front());
chain.push_back(ret);
++allocated;
}

if(allocated != n){
mp_node_pool->allocate_nodes(n - allocated, chain);
}
}
BOOST_CATCH(...){
this->cached_deallocation(chain);
BOOST_RETHROW
}
BOOST_CATCH_END
}

void cached_deallocation(void *ptr)
{
if(m_cached_nodes.size() >= m_max_cached_nodes){
this->priv_deallocate_n_nodes(m_cached_nodes.size() - m_max_cached_nodes/2);
}
m_cached_nodes.push_front(ptr);
}

void cached_deallocation(multiallocation_chain &chain)
{
m_cached_nodes.splice_after(m_cached_nodes.before_begin(), chain);

if(m_cached_nodes.size() >= m_max_cached_nodes){
this->priv_deallocate_n_nodes(m_cached_nodes.size() - m_max_cached_nodes/2);
}
}

void set_max_cached_nodes(size_type newmax)
{
m_max_cached_nodes = newmax;
this->priv_deallocate_remaining_nodes();
}

void deallocate_all_cached_nodes()
{
if(m_cached_nodes.empty()) return;
mp_node_pool->deallocate_nodes(m_cached_nodes);
}

private:
void priv_deallocate_remaining_nodes()
{
if(m_cached_nodes.size() > m_max_cached_nodes){
priv_deallocate_n_nodes(m_cached_nodes.size()-m_max_cached_nodes);
}
}

void priv_deallocate_n_nodes(size_type n)
{
size_type count(n);
typename multiallocation_chain::iterator it(m_cached_nodes.before_begin());
while(count--){
++it;
}
multiallocation_chain chain;
chain.splice_after(chain.before_begin(), m_cached_nodes, m_cached_nodes.before_begin(), it, n);
mp_node_pool->deallocate_nodes(chain);
}

public:
void swap(cache_impl &other)
{
::boost::adl_move_swap(mp_node_pool, other.mp_node_pool);
::boost::adl_move_swap(m_cached_nodes, other.m_cached_nodes);
::boost::adl_move_swap(m_max_cached_nodes, other.m_max_cached_nodes);
}
};

template<class Derived, class T, class SegmentManager>
class array_allocation_impl
{
const Derived *derived() const
{  return static_cast<const Derived*>(this); }
Derived *derived()
{  return static_cast<Derived*>(this); }

typedef typename SegmentManager::void_pointer         void_pointer;

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
typedef typename SegmentManager::size_type            size_type;
typedef typename SegmentManager::difference_type      difference_type;
typedef boost::container::dtl::transform_multiallocation_chain
<typename SegmentManager::multiallocation_chain, T>multiallocation_chain;


public:
size_type size(const pointer &p) const
{
return (size_type)this->derived()->get_segment_manager()->size(ipcdetail::to_raw_pointer(p))/sizeof(T);
}

pointer allocation_command(boost::interprocess::allocation_type command,
size_type limit_size, size_type &prefer_in_recvd_out_size, pointer &reuse)
{
value_type *reuse_raw = ipcdetail::to_raw_pointer(reuse);
pointer const p = this->derived()->get_segment_manager()->allocation_command
(command, limit_size, prefer_in_recvd_out_size, reuse_raw);
reuse = reuse_raw;
return p;
}

void allocate_many(size_type elem_size, size_type num_elements, multiallocation_chain &chain)
{
if(size_overflows<sizeof(T)>(elem_size)){
throw bad_alloc();
}
this->derived()->get_segment_manager()->allocate_many(elem_size*sizeof(T), num_elements, chain);
}

void allocate_many(const size_type *elem_sizes, size_type n_elements, multiallocation_chain &chain)
{
this->derived()->get_segment_manager()->allocate_many(elem_sizes, n_elements, sizeof(T), chain);
}

void deallocate_many(multiallocation_chain &chain)
{  this->derived()->get_segment_manager()->deallocate_many(chain); }

size_type max_size() const
{  return this->derived()->get_segment_manager()->get_size()/sizeof(T);  }

pointer address(reference value) const
{  return pointer(boost::addressof(value));  }

const_pointer address(const_reference value) const
{  return const_pointer(boost::addressof(value));  }

template<class P>
void construct(const pointer &ptr, BOOST_FWD_REF(P) p)
{  ::new((void*)ipcdetail::to_raw_pointer(ptr), boost_container_new_t()) value_type(::boost::forward<P>(p));  }

void destroy(const pointer &ptr)
{  BOOST_ASSERT(ptr != 0); (*ptr).~value_type();  }
};


template<class Derived, unsigned int Version, class T, class SegmentManager>
class node_pool_allocation_impl
:  public array_allocation_impl
< Derived
, T
, SegmentManager>
{
const Derived *derived() const
{  return static_cast<const Derived*>(this); }
Derived *derived()
{  return static_cast<Derived*>(this); }

typedef typename SegmentManager::void_pointer         void_pointer;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<const void>::type                cvoid_pointer;

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
typedef typename SegmentManager::size_type            size_type;
typedef typename SegmentManager::difference_type      difference_type;
typedef boost::container::dtl::transform_multiallocation_chain
<typename SegmentManager::multiallocation_chain, T>multiallocation_chain;


template <int Dummy>
struct node_pool
{
typedef typename Derived::template node_pool<0>::type type;
static type *get(void *p)
{  return static_cast<type*>(p); }
};

public:
pointer allocate(size_type count, cvoid_pointer hint = 0)
{
(void)hint;
typedef typename node_pool<0>::type node_pool_t;
node_pool_t *pool = node_pool<0>::get(this->derived()->get_node_pool());
if(size_overflows<sizeof(T)>(count)){
throw bad_alloc();
}
else if(Version == 1 && count == 1){
return pointer(static_cast<value_type*>
(pool->allocate_node()));
}
else{
return pointer(static_cast<value_type*>
(pool->get_segment_manager()->allocate(count*sizeof(T))));
}
}

void deallocate(const pointer &ptr, size_type count)
{
(void)count;
typedef typename node_pool<0>::type node_pool_t;
node_pool_t *pool = node_pool<0>::get(this->derived()->get_node_pool());
if(Version == 1 && count == 1)
pool->deallocate_node(ipcdetail::to_raw_pointer(ptr));
else
pool->get_segment_manager()->deallocate((void*)ipcdetail::to_raw_pointer(ptr));
}

pointer allocate_one()
{
typedef typename node_pool<0>::type node_pool_t;
node_pool_t *pool = node_pool<0>::get(this->derived()->get_node_pool());
return pointer(static_cast<value_type*>(pool->allocate_node()));
}

void allocate_individual(size_type num_elements, multiallocation_chain &chain)
{
typedef typename node_pool<0>::type node_pool_t;
node_pool_t *pool = node_pool<0>::get(this->derived()->get_node_pool());
pool->allocate_nodes(num_elements, chain);
}

void deallocate_one(const pointer &p)
{
typedef typename node_pool<0>::type node_pool_t;
node_pool_t *pool = node_pool<0>::get(this->derived()->get_node_pool());
pool->deallocate_node(ipcdetail::to_raw_pointer(p));
}

void deallocate_individual(multiallocation_chain &chain)
{
node_pool<0>::get(this->derived()->get_node_pool())->deallocate_nodes
(chain);
}

void deallocate_free_blocks()
{  node_pool<0>::get(this->derived()->get_node_pool())->deallocate_free_blocks();  }

void deallocate_free_chunks()
{  node_pool<0>::get(this->derived()->get_node_pool())->deallocate_free_blocks();  }
};

template<class T, class NodePool, unsigned int Version>
class cached_allocator_impl
:  public array_allocation_impl
<cached_allocator_impl<T, NodePool, Version>, T, typename NodePool::segment_manager>
{
cached_allocator_impl & operator=(const cached_allocator_impl& other);
typedef array_allocation_impl
< cached_allocator_impl
<T, NodePool, Version>
, T
, typename NodePool::segment_manager> base_t;

public:
typedef NodePool                                      node_pool_t;
typedef typename NodePool::segment_manager            segment_manager;
typedef typename segment_manager::void_pointer        void_pointer;
typedef typename boost::intrusive::
pointer_traits<void_pointer>::template
rebind_pointer<const void>::type                cvoid_pointer;
typedef typename base_t::pointer                      pointer;
typedef typename base_t::size_type                    size_type;
typedef typename base_t::multiallocation_chain        multiallocation_chain;
typedef typename base_t::value_type                   value_type;

public:
static const std::size_t DEFAULT_MAX_CACHED_NODES = 64;

cached_allocator_impl(segment_manager *segment_mngr, size_type max_cached_nodes)
: m_cache(segment_mngr, max_cached_nodes)
{}

cached_allocator_impl(const cached_allocator_impl &other)
: m_cache(other.m_cache)
{}

template<class T2, class NodePool2>
cached_allocator_impl
(const cached_allocator_impl
<T2, NodePool2, Version> &other)
: m_cache(other.get_segment_manager(), other.get_max_cached_nodes())
{}

node_pool_t* get_node_pool() const
{  return m_cache.get_node_pool();   }

segment_manager* get_segment_manager()const
{  return m_cache.get_segment_manager();   }

void set_max_cached_nodes(size_type newmax)
{  m_cache.set_max_cached_nodes(newmax);   }

size_type get_max_cached_nodes() const
{  return m_cache.get_max_cached_nodes();   }

pointer allocate(size_type count, cvoid_pointer hint = 0)
{
(void)hint;
void * ret;
if(size_overflows<sizeof(T)>(count)){
throw bad_alloc();
}
else if(Version == 1 && count == 1){
ret = m_cache.cached_allocation();
}
else{
ret = this->get_segment_manager()->allocate(count*sizeof(T));
}
return pointer(static_cast<T*>(ret));
}

void deallocate(const pointer &ptr, size_type count)
{
(void)count;
if(Version == 1 && count == 1){
m_cache.cached_deallocation(ipcdetail::to_raw_pointer(ptr));
}
else{
this->get_segment_manager()->deallocate((void*)ipcdetail::to_raw_pointer(ptr));
}
}

pointer allocate_one()
{  return pointer(static_cast<value_type*>(this->m_cache.cached_allocation()));   }

void allocate_individual(size_type num_elements, multiallocation_chain &chain)
{  this->m_cache.cached_allocation(num_elements, chain);   }

void deallocate_one(const pointer &p)
{  this->m_cache.cached_deallocation(ipcdetail::to_raw_pointer(p)); }

void deallocate_individual(multiallocation_chain &chain)
{  m_cache.cached_deallocation(chain);  }

void deallocate_free_blocks()
{  m_cache.get_node_pool()->deallocate_free_blocks();   }

friend void swap(cached_allocator_impl &alloc1, cached_allocator_impl &alloc2)
{  ::boost::adl_move_swap(alloc1.m_cache, alloc2.m_cache);   }

void deallocate_cache()
{  m_cache.deallocate_all_cached_nodes(); }

void deallocate_free_chunks()
{  m_cache.get_node_pool()->deallocate_free_blocks();   }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
cache_impl<node_pool_t> m_cache;
#endif   
};

template<class T, class N, unsigned int V> inline
bool operator==(const cached_allocator_impl<T, N, V> &alloc1,
const cached_allocator_impl<T, N, V> &alloc2)
{  return alloc1.get_node_pool() == alloc2.get_node_pool(); }

template<class T, class N, unsigned int V> inline
bool operator!=(const cached_allocator_impl<T, N, V> &alloc1,
const cached_allocator_impl<T, N, V> &alloc2)
{  return alloc1.get_node_pool() != alloc2.get_node_pool(); }


template<class private_node_allocator_t>
class shared_pool_impl
: public private_node_allocator_t
{
public:
typedef typename private_node_allocator_t::
segment_manager                           segment_manager;
typedef typename private_node_allocator_t::
multiallocation_chain                     multiallocation_chain;
typedef typename private_node_allocator_t::
size_type                                 size_type;

private:
typedef typename segment_manager::mutex_family::mutex_type mutex_type;

public:
shared_pool_impl(segment_manager *segment_mngr)
: private_node_allocator_t(segment_mngr)
{}

~shared_pool_impl()
{}

void *allocate_node()
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
return private_node_allocator_t::allocate_node();
}

void deallocate_node(void *ptr)
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::deallocate_node(ptr);
}

void allocate_nodes(const size_type n, multiallocation_chain &chain)
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::allocate_nodes(n, chain);
}

void deallocate_nodes(multiallocation_chain &nodes, size_type num)
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::deallocate_nodes(nodes, num);
}

void deallocate_nodes(multiallocation_chain &chain)
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::deallocate_nodes(chain);
}

void deallocate_free_blocks()
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::deallocate_free_blocks();
}

void purge_blocks()
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::purge_blocks();
}

size_type inc_ref_count()
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
return ++m_header.m_usecount;
}

size_type dec_ref_count()
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
BOOST_ASSERT(m_header.m_usecount > 0);
return --m_header.m_usecount;
}

void deallocate_free_chunks()
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::deallocate_free_blocks();
}

void purge_chunks()
{
boost::interprocess::scoped_lock<mutex_type> guard(m_header);
private_node_allocator_t::purge_blocks();
}

private:
struct header_t : mutex_type
{
size_type m_usecount;    

header_t()
:  m_usecount(0) {}
} m_header;
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
