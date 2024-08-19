#ifndef BOOST_CONTAINER_DETAIL_NODE_POOL_IMPL_HPP
#define BOOST_CONTAINER_DETAIL_NODE_POOL_IMPL_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/container_fwd.hpp>

#include <boost/container/detail/math_functions.hpp>
#include <boost/container/detail/mpl.hpp>
#include <boost/container/detail/pool_common.hpp>
#include <boost/move/detail/to_raw_pointer.hpp>
#include <boost/container/detail/type_traits.hpp>

#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/set.hpp>
#include <boost/intrusive/slist.hpp>

#include <boost/core/no_exceptions_support.hpp>
#include <boost/assert.hpp>
#include <cstddef>

namespace boost {
namespace container {
namespace dtl {

template<class SegmentManagerBase>
class private_node_pool_impl
{
private_node_pool_impl();
private_node_pool_impl(const private_node_pool_impl &);
private_node_pool_impl &operator=(const private_node_pool_impl &);

public:
typedef typename SegmentManagerBase::void_pointer              void_pointer;
typedef typename node_slist<void_pointer>::slist_hook_t        slist_hook_t;
typedef typename node_slist<void_pointer>::node_t              node_t;
typedef typename node_slist<void_pointer>::node_slist_t        free_nodes_t;
typedef typename SegmentManagerBase::multiallocation_chain     multiallocation_chain;
typedef typename SegmentManagerBase::size_type                 size_type;

private:
typedef typename bi::make_slist
< node_t, bi::base_hook<slist_hook_t>
, bi::linear<true>
, bi::constant_time_size<false> >::type      blockslist_t;

static size_type get_rounded_size(size_type orig_size, size_type round_to)
{  return ((orig_size-1)/round_to+1)*round_to;  }

public:

typedef SegmentManagerBase segment_manager_base_type;

private_node_pool_impl(segment_manager_base_type *segment_mngr_base, size_type node_size, size_type nodes_per_block)
:  m_nodes_per_block(nodes_per_block)
,  m_real_node_size(lcm(node_size, size_type(alignment_of<node_t>::value)))
,  mp_segment_mngr_base(segment_mngr_base)
,  m_blocklist()
,  m_freelist()
,  m_allocated(0)
{}

~private_node_pool_impl()
{  this->purge_blocks();  }

size_type get_real_num_node() const
{  return m_nodes_per_block; }

segment_manager_base_type* get_segment_manager_base()const
{  return boost::movelib::to_raw_pointer(mp_segment_mngr_base);  }

void *allocate_node()
{  return this->priv_alloc_node();  }

void deallocate_node(void *ptr)
{  this->priv_dealloc_node(ptr); }

void allocate_nodes(const size_type n, multiallocation_chain &chain)
{
size_type cur_nodes = m_freelist.size();
if(cur_nodes < n){
this->priv_alloc_block(((n - cur_nodes) - 1)/m_nodes_per_block + 1);
}

typedef typename free_nodes_t::iterator free_iterator;
free_iterator before_last_new_it = m_freelist.before_begin();
for(size_type j = 0; j != n; ++j){
++before_last_new_it;
}

free_iterator first_node(m_freelist.begin());
free_iterator last_node (before_last_new_it);

m_freelist.erase_after( m_freelist.before_begin()
, ++free_iterator(before_last_new_it)
, n);

chain.incorporate_after(chain.before_begin(), &*first_node, &*last_node, n);
m_allocated += n;
}

void deallocate_nodes(multiallocation_chain &chain)
{
typedef typename multiallocation_chain::iterator iterator;
iterator it(chain.begin()), itend(chain.end());
while(it != itend){
void *pElem = &*it;
++it;
this->priv_dealloc_node(pElem);
}
}

void deallocate_free_blocks()
{
typedef typename free_nodes_t::iterator nodelist_iterator;
typename blockslist_t::iterator bit(m_blocklist.before_begin()),
it(m_blocklist.begin()),
itend(m_blocklist.end());
free_nodes_t backup_list;
nodelist_iterator backup_list_last = backup_list.before_begin();

size_type blocksize = (get_rounded_size)
(m_real_node_size*m_nodes_per_block, (size_type) alignment_of<node_t>::value);

while(it != itend){
free_nodes_t free_nodes;
nodelist_iterator last_it = free_nodes.before_begin();
const void *addr = get_block_from_hook(&*it, blocksize);

m_freelist.remove_and_dispose_if
(is_between(addr, blocksize), push_in_list(free_nodes, last_it));

if(free_nodes.size() == m_nodes_per_block){
free_nodes.clear();
it = m_blocklist.erase_after(bit);
mp_segment_mngr_base->deallocate((void*)addr);
}
else{
if(backup_list.empty() && !m_freelist.empty()){
backup_list_last = last_it;
}
backup_list.splice_after
( backup_list.before_begin()
, free_nodes
, free_nodes.before_begin()
, last_it
, free_nodes.size());
bit = it;
++it;
}
}
BOOST_ASSERT(m_freelist.empty());

m_freelist.splice_after
( m_freelist.before_begin()
, backup_list
, backup_list.before_begin()
, backup_list_last
, backup_list.size());
}

size_type num_free_nodes()
{  return m_freelist.size();  }

void purge_blocks()
{
BOOST_ASSERT(m_allocated==0);
size_type blocksize = (get_rounded_size)
(m_real_node_size*m_nodes_per_block, (size_type)alignment_of<node_t>::value);

while(!m_blocklist.empty()){
void *addr = get_block_from_hook(&m_blocklist.front(), blocksize);
m_blocklist.pop_front();
mp_segment_mngr_base->deallocate((void*)addr);
}
m_freelist.clear();
}

void swap(private_node_pool_impl &other)
{
BOOST_ASSERT(m_nodes_per_block == other.m_nodes_per_block);
BOOST_ASSERT(m_real_node_size == other.m_real_node_size);
std::swap(mp_segment_mngr_base, other.mp_segment_mngr_base);
m_blocklist.swap(other.m_blocklist);
m_freelist.swap(other.m_freelist);
std::swap(m_allocated, other.m_allocated);
}

private:

struct push_in_list
{
push_in_list(free_nodes_t &l, typename free_nodes_t::iterator &it)
:  slist_(l), last_it_(it)
{}

void operator()(typename free_nodes_t::pointer p) const
{
slist_.push_front(*p);
if(slist_.size() == 1){ 
++last_it_ = slist_.begin();
}
}

private:
free_nodes_t &slist_;
typename free_nodes_t::iterator &last_it_;
};

struct is_between
{
typedef typename free_nodes_t::value_type argument_type;
typedef bool                              result_type;

is_between(const void *addr, std::size_t size)
:  beg_(static_cast<const char *>(addr)), end_(beg_+size)
{}

bool operator()(typename free_nodes_t::const_reference v) const
{
return (beg_ <= reinterpret_cast<const char *>(&v) &&
end_ >  reinterpret_cast<const char *>(&v));
}
private:
const char *      beg_;
const char *      end_;
};

node_t *priv_alloc_node()
{
if (m_freelist.empty())
this->priv_alloc_block(1);
node_t *n = (node_t*)&m_freelist.front();
m_freelist.pop_front();
++m_allocated;
return n;
}

void priv_dealloc_node(void *pElem)
{
node_t * to_deallocate = static_cast<node_t*>(pElem);
m_freelist.push_front(*to_deallocate);
BOOST_ASSERT(m_allocated>0);
--m_allocated;
}

void priv_alloc_block(size_type num_blocks)
{
BOOST_ASSERT(num_blocks > 0);
size_type blocksize =
(get_rounded_size)(m_real_node_size*m_nodes_per_block, (size_type)alignment_of<node_t>::value);

BOOST_TRY{
for(size_type i = 0; i != num_blocks; ++i){
char *pNode = reinterpret_cast<char*>
(mp_segment_mngr_base->allocate(blocksize + sizeof(node_t)));
char *pBlock = pNode;
m_blocklist.push_front(get_block_hook(pBlock, blocksize));

for(size_type j = 0; j < m_nodes_per_block; ++j, pNode += m_real_node_size){
m_freelist.push_front(*new (pNode) node_t);
}
}
}
BOOST_CATCH(...){
BOOST_RETHROW
}
BOOST_CATCH_END
}

void deallocate_free_chunks()
{  this->deallocate_free_blocks(); }

void purge_chunks()
{  this->purge_blocks(); }

private:
static node_t & get_block_hook (void *block, size_type blocksize)
{
return *reinterpret_cast<node_t*>(reinterpret_cast<char*>(block) + blocksize);
}

void *get_block_from_hook (node_t *hook, size_type blocksize)
{
return (reinterpret_cast<char*>(hook) - blocksize);
}

private:
typedef typename boost::intrusive::pointer_traits
<void_pointer>::template rebind_pointer<segment_manager_base_type>::type   segment_mngr_base_ptr_t;

const size_type m_nodes_per_block;
const size_type m_real_node_size;
segment_mngr_base_ptr_t mp_segment_mngr_base;   
blockslist_t      m_blocklist;      
free_nodes_t      m_freelist;       
size_type       m_allocated;      
};


}  
}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
