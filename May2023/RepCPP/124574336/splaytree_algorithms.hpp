
#ifndef BOOST_INTRUSIVE_SPLAYTREE_ALGORITHMS_HPP
#define BOOST_INTRUSIVE_SPLAYTREE_ALGORITHMS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/assert.hpp>
#include <boost/intrusive/detail/algo_type.hpp>
#include <boost/intrusive/detail/uncast.hpp>
#include <boost/intrusive/bstree_algorithms.hpp>

#include <cstddef>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

namespace detail {

template<class NodeTraits>
struct splaydown_assemble_and_fix_header
{
typedef typename NodeTraits::node_ptr node_ptr;

splaydown_assemble_and_fix_header(node_ptr t, node_ptr header, node_ptr leftmost, node_ptr rightmost)
: t_(t)
, null_node_(header)
, l_(null_node_)
, r_(null_node_)
, leftmost_(leftmost)
, rightmost_(rightmost)
{}

~splaydown_assemble_and_fix_header()
{
this->assemble();

NodeTraits::set_parent(null_node_, t_);
NodeTraits::set_parent(t_, null_node_);
NodeTraits::set_left (null_node_, leftmost_);
NodeTraits::set_right(null_node_, rightmost_);
}

private:

void assemble()
{
{  

node_ptr const old_t_left  = NodeTraits::get_left(t_);
node_ptr const old_t_right = NodeTraits::get_right(t_);
NodeTraits::set_right(l_, old_t_left);
NodeTraits::set_left (r_, old_t_right);
if(old_t_left){
NodeTraits::set_parent(old_t_left, l_);
}
if(old_t_right){
NodeTraits::set_parent(old_t_right, r_);
}
}
{  
node_ptr const null_right = NodeTraits::get_right(null_node_);
node_ptr const null_left  = NodeTraits::get_left(null_node_);
NodeTraits::set_left (t_, null_right);
NodeTraits::set_right(t_, null_left);
if(null_right){
NodeTraits::set_parent(null_right, t_);
}
if(null_left){
NodeTraits::set_parent(null_left, t_);
}
}
}

public:
node_ptr t_, null_node_, l_, r_, leftmost_, rightmost_;
};

}  

template<class NodeTraits>
class splaytree_algorithms
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
: public bstree_algorithms<NodeTraits>
#endif
{
private:
typedef bstree_algorithms<NodeTraits> bstree_algo;

public:
typedef typename NodeTraits::node            node;
typedef NodeTraits                           node_traits;
typedef typename NodeTraits::node_ptr        node_ptr;
typedef typename NodeTraits::const_node_ptr  const_node_ptr;

typedef typename bstree_algo::insert_commit_data insert_commit_data;

public:
#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
static node_ptr get_header(const const_node_ptr & n);

static node_ptr begin_node(const const_node_ptr & header);

static node_ptr end_node(const const_node_ptr & header);

static void swap_tree(const node_ptr & header1, const node_ptr & header2);

static void swap_nodes(node_ptr node1, node_ptr node2);

static void swap_nodes(node_ptr node1, node_ptr header1, node_ptr node2, node_ptr header2);

static void replace_node(node_ptr node_to_be_replaced, node_ptr new_node);

static void replace_node(node_ptr node_to_be_replaced, node_ptr header, node_ptr new_node);

static void unlink(node_ptr node);

static node_ptr unlink_leftmost_without_rebalance(node_ptr header);

static bool unique(const_node_ptr node);

static std::size_t size(const_node_ptr header);

static node_ptr next_node(node_ptr node);

static node_ptr prev_node(node_ptr node);

static void init(node_ptr node);

static void init_header(node_ptr header);

#endif   

static void erase(node_ptr header, node_ptr z)
{
if(NodeTraits::get_left(z)){
splay_up(bstree_algo::prev_node(z), header);
}




bstree_algo::erase(header, z);
}

template<class NodePtrCompare>
static bool transfer_unique
(node_ptr header1, NodePtrCompare comp, node_ptr header2, node_ptr z)
{
typename bstree_algo::insert_commit_data commit_data;
bool const transferable = bstree_algo::insert_unique_check(header1, z, comp, commit_data).second;
if(transferable){
erase(header2, z);
bstree_algo::insert_commit(header1, z, commit_data);
splay_up(z, header1);
}
return transferable;
}

template<class NodePtrCompare>
static void transfer_equal
(node_ptr header1, NodePtrCompare comp, node_ptr header2, node_ptr z)
{
insert_commit_data commit_data;
splay_down(header1, z, comp);
bstree_algo::insert_equal_upper_bound_check(header1, z, comp, commit_data);
erase(header2, z);
bstree_algo::insert_commit(header1, z, commit_data);
}

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
template <class Cloner, class Disposer>
static void clone
(const_node_ptr source_header, node_ptr target_header, Cloner cloner, Disposer disposer);

template<class Disposer>
static void clear_and_dispose(node_ptr header, Disposer disposer);

#endif   
template<class KeyType, class KeyNodePtrCompare>
static std::size_t count
(node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{
std::pair<node_ptr, node_ptr> ret = equal_range(header, key, comp);
std::size_t n = 0;
while(ret.first != ret.second){
++n;
ret.first = next_node(ret.first);
}
return n;
}

template<class KeyType, class KeyNodePtrCompare>
static std::size_t count
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{  return bstree_algo::count(header, key, comp);  }

template<class KeyType, class KeyNodePtrCompare>
static node_ptr lower_bound
(node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{
splay_down(detail::uncast(header), key, comp);
node_ptr y = bstree_algo::lower_bound(header, key, comp);
return y;
}

template<class KeyType, class KeyNodePtrCompare>
static node_ptr lower_bound
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{  return bstree_algo::lower_bound(header, key, comp);  }

template<class KeyType, class KeyNodePtrCompare>
static node_ptr upper_bound
(node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{
splay_down(detail::uncast(header), key, comp);
node_ptr y = bstree_algo::upper_bound(header, key, comp);
return y;
}

template<class KeyType, class KeyNodePtrCompare>
static node_ptr upper_bound
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{  return bstree_algo::upper_bound(header, key, comp);  }

template<class KeyType, class KeyNodePtrCompare>
static node_ptr find
(node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{
splay_down(detail::uncast(header), key, comp);
return bstree_algo::find(header, key, comp);
}

template<class KeyType, class KeyNodePtrCompare>
static node_ptr find
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{  return bstree_algo::find(header, key, comp);  }

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> equal_range
(node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{
splay_down(detail::uncast(header), key, comp);
std::pair<node_ptr, node_ptr> ret = bstree_algo::equal_range(header, key, comp);
return ret;
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> equal_range
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{  return bstree_algo::equal_range(header, key, comp);  }

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> lower_bound_range
(node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{
splay_down(detail::uncast(header), key, comp);
std::pair<node_ptr, node_ptr> ret = bstree_algo::lower_bound_range(header, key, comp);
return ret;
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> lower_bound_range
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp)
{  return bstree_algo::lower_bound_range(header, key, comp);  }

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> bounded_range
(node_ptr header, const KeyType &lower_key, const KeyType &upper_key, KeyNodePtrCompare comp
, bool left_closed, bool right_closed)
{
splay_down(detail::uncast(header), lower_key, comp);
std::pair<node_ptr, node_ptr> ret =
bstree_algo::bounded_range(header, lower_key, upper_key, comp, left_closed, right_closed);
return ret;
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> bounded_range
(const_node_ptr header, const KeyType &lower_key, const KeyType &upper_key, KeyNodePtrCompare comp
, bool left_closed, bool right_closed)
{  return bstree_algo::bounded_range(header, lower_key, upper_key, comp, left_closed, right_closed);  }

template<class NodePtrCompare>
static node_ptr insert_equal_upper_bound
(node_ptr header, node_ptr new_node, NodePtrCompare comp)
{
splay_down(header, new_node, comp);
return bstree_algo::insert_equal_upper_bound(header, new_node, comp);
}

template<class NodePtrCompare>
static node_ptr insert_equal_lower_bound
(node_ptr header, node_ptr new_node, NodePtrCompare comp)
{
splay_down(header, new_node, comp);
return bstree_algo::insert_equal_lower_bound(header, new_node, comp);
}  

template<class NodePtrCompare>
static node_ptr insert_equal
(node_ptr header, node_ptr hint, node_ptr new_node, NodePtrCompare comp)
{
splay_down(header, new_node, comp);
return bstree_algo::insert_equal(header, hint, new_node, comp);
}

static node_ptr insert_before
(node_ptr header, node_ptr pos, node_ptr new_node)
{
bstree_algo::insert_before(header, pos, new_node);
splay_up(new_node, header);
return new_node;
}

static void push_back(node_ptr header, node_ptr new_node)
{
bstree_algo::push_back(header, new_node);
splay_up(new_node, header);
}

static void push_front(node_ptr header, node_ptr new_node)
{
bstree_algo::push_front(header, new_node);
splay_up(new_node, header);
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, bool> insert_unique_check
(node_ptr header, const KeyType &key
,KeyNodePtrCompare comp, insert_commit_data &commit_data)
{
splay_down(header, key, comp);
return bstree_algo::insert_unique_check(header, key, comp, commit_data);
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, bool> insert_unique_check
(node_ptr header, node_ptr hint, const KeyType &key
,KeyNodePtrCompare comp, insert_commit_data &commit_data)
{
splay_down(header, key, comp);
return bstree_algo::insert_unique_check(header, hint, key, comp, commit_data);
}

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
static void insert_unique_commit
(node_ptr header, node_ptr new_value, const insert_commit_data &commit_data);

static bool is_header(const_node_ptr p);

static void rebalance(node_ptr header);

static node_ptr rebalance_subtree(node_ptr old_root);

#endif   

static void splay_up(node_ptr node, node_ptr header)
{  priv_splay_up<true>(node, header); }

template<class KeyType, class KeyNodePtrCompare>
static node_ptr splay_down(node_ptr header, const KeyType &key, KeyNodePtrCompare comp, bool *pfound = 0)
{  return priv_splay_down<true>(header, key, comp, pfound);   }

private:


template<bool SimpleSplay>
static void priv_splay_up(node_ptr node, node_ptr header)
{
node_ptr n((node == header) ? NodeTraits::get_right(header) : node);
node_ptr t(header);

if( n == t ) return;

for( ;; ){
node_ptr p(NodeTraits::get_parent(n));
node_ptr g(NodeTraits::get_parent(p));

if( p == t )   break;

if( g == t ){
rotate(n);
}
else if ((NodeTraits::get_left(p) == n && NodeTraits::get_left(g) == p)    ||
(NodeTraits::get_right(p) == n && NodeTraits::get_right(g) == p)  ){
rotate(p);
rotate(n);
}
else {
rotate(n);
if(!SimpleSplay){
rotate(n);
}
}
}
}

template<bool SimpleSplay, class KeyType, class KeyNodePtrCompare>
static node_ptr priv_splay_down(node_ptr header, const KeyType &key, KeyNodePtrCompare comp, bool *pfound = 0)
{
node_ptr const old_root  = NodeTraits::get_parent(header);
node_ptr const leftmost  = NodeTraits::get_left(header);
node_ptr const rightmost = NodeTraits::get_right(header);
if(leftmost == rightmost){ 
if(pfound){
*pfound = old_root && !comp(key, old_root) && !comp(old_root, key);
}
return old_root ? old_root : header;
}
else{
NodeTraits::set_left (header, node_ptr());
NodeTraits::set_right(header, node_ptr());
detail::splaydown_assemble_and_fix_header<NodeTraits> commit(old_root, header, leftmost, rightmost);
bool found = false;

for( ;; ){
if(comp(key, commit.t_)){
node_ptr const t_left = NodeTraits::get_left(commit.t_);
if(!t_left)
break;
if(comp(key, t_left)){
bstree_algo::rotate_right_no_parent_fix(commit.t_, t_left);
commit.t_ = t_left;
if( !NodeTraits::get_left(commit.t_) )
break;
link_right(commit.t_, commit.r_);
}
else{
link_right(commit.t_, commit.r_);
if(!SimpleSplay && comp(t_left, key)){
if( !NodeTraits::get_right(commit.t_) )
break;
link_left(commit.t_, commit.l_);
}
}
}
else if(comp(commit.t_, key)){
node_ptr const t_right = NodeTraits::get_right(commit.t_);
if(!t_right)
break;

if(comp(t_right, key)){
bstree_algo::rotate_left_no_parent_fix(commit.t_, t_right);
commit.t_ = t_right;
if( !NodeTraits::get_right(commit.t_) )
break;
link_left(commit.t_, commit.l_);
}
else{
link_left(commit.t_, commit.l_);
if(!SimpleSplay && comp(key, t_right)){
if( !NodeTraits::get_left(commit.t_) )
break;
link_right(commit.t_, commit.r_);
}
}
}
else{
found = true;
break;
}
}

if(pfound){
*pfound = found;
}
return commit.t_;
}
}

static void link_left(node_ptr & t, node_ptr & l)
{
NodeTraits::set_right(l, t);
NodeTraits::set_parent(t, l);
l = t;
t = NodeTraits::get_right(t);
}

static void link_right(node_ptr & t, node_ptr & r)
{
NodeTraits::set_left(r, t);
NodeTraits::set_parent(t, r);
r = t;
t = NodeTraits::get_left(t);
}

static void rotate(node_ptr n)
{
node_ptr p = NodeTraits::get_parent(n);
node_ptr g = NodeTraits::get_parent(p);
bool g_is_header = bstree_algo::is_header(g);

if(NodeTraits::get_left(p) == n){
NodeTraits::set_left(p, NodeTraits::get_right(n));
if(NodeTraits::get_left(p))
NodeTraits::set_parent(NodeTraits::get_left(p), p);
NodeTraits::set_right(n, p);
}
else{ 
NodeTraits::set_right(p, NodeTraits::get_left(n));
if(NodeTraits::get_right(p))
NodeTraits::set_parent(NodeTraits::get_right(p), p);
NodeTraits::set_left(n, p);
}

NodeTraits::set_parent(p, n);
NodeTraits::set_parent(n, g);

if(g_is_header){
if(NodeTraits::get_parent(g) == p)
NodeTraits::set_parent(g, n);
else{
BOOST_INTRUSIVE_INVARIANT_ASSERT(false);
NodeTraits::set_right(g, n);
}
}
else{
if(NodeTraits::get_left(g) == p)
NodeTraits::set_left(g, n);
else  
NodeTraits::set_right(g, n);
}
}

};


template<class NodeTraits>
struct get_algo<SplayTreeAlgorithms, NodeTraits>
{
typedef splaytree_algorithms<NodeTraits> type;
};

template <class ValueTraits, class NodePtrCompare, class ExtraChecker>
struct get_node_checker<SplayTreeAlgorithms, ValueTraits, NodePtrCompare, ExtraChecker>
{
typedef detail::bstree_node_checker<ValueTraits, NodePtrCompare, ExtraChecker> type;
};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
