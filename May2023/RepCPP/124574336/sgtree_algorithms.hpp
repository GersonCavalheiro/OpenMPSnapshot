#ifndef BOOST_INTRUSIVE_SGTREE_ALGORITHMS_HPP
#define BOOST_INTRUSIVE_SGTREE_ALGORITHMS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <cstddef>
#include <boost/intrusive/detail/algo_type.hpp>
#include <boost/intrusive/bstree_algorithms.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template<class NodeTraits>
class sgtree_algorithms
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
: public bstree_algorithms<NodeTraits>
#endif
{
public:
typedef typename NodeTraits::node            node;
typedef NodeTraits                           node_traits;
typedef typename NodeTraits::node_ptr       node_ptr;
typedef typename NodeTraits::const_node_ptr const_node_ptr;

private:

typedef bstree_algorithms<NodeTraits>  bstree_algo;


public:
struct insert_commit_data
: bstree_algo::insert_commit_data
{
std::size_t depth;
};

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
static node_ptr get_header(const_node_ptr n);

static node_ptr begin_node(const_node_ptr header);

static node_ptr end_node(const_node_ptr header);

static void swap_tree(node_ptr header1, node_ptr header2);

static void swap_nodes(node_ptr node1, node_ptr node2);

static void swap_nodes(node_ptr node1, node_ptr header1, node_ptr node2, node_ptr header2);

static void replace_node(node_ptr node_to_be_replaced, node_ptr new_node);

static void replace_node(node_ptr node_to_be_replaced, node_ptr header, node_ptr new_node);


static node_ptr unlink_leftmost_without_rebalance(node_ptr header);

static bool unique(const_node_ptr node);

static std::size_t size(const_node_ptr header);

static node_ptr next_node(node_ptr node);

static node_ptr prev_node(node_ptr node);

static void init(node_ptr node);

static void init_header(node_ptr header);
#endif   

template<class AlphaByMaxSize>
static node_ptr erase(node_ptr header, node_ptr z, std::size_t tree_size, std::size_t &max_tree_size, AlphaByMaxSize alpha_by_maxsize)
{
bstree_algo::erase(header, z);
--tree_size;
if (tree_size > 0 &&
tree_size < alpha_by_maxsize(max_tree_size)){
bstree_algo::rebalance(header);
max_tree_size = tree_size;
}
return z;
}

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
template <class Cloner, class Disposer>
static void clone
(const_node_ptr source_header, node_ptr target_header, Cloner cloner, Disposer disposer);

template<class Disposer>
static void clear_and_dispose(node_ptr header, Disposer disposer);

template<class KeyType, class KeyNodePtrCompare>
static node_ptr lower_bound
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp);

template<class KeyType, class KeyNodePtrCompare>
static node_ptr upper_bound
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp);

template<class KeyType, class KeyNodePtrCompare>
static node_ptr find
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp);

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> equal_range
(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp);

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> bounded_range
(const_node_ptr header, const KeyType &lower_key, const KeyType &upper_key, KeyNodePtrCompare comp
, bool left_closed, bool right_closed);

template<class KeyType, class KeyNodePtrCompare>
static std::size_t count(const_node_ptr header, const KeyType &key, KeyNodePtrCompare comp);

#endif   

template<class NodePtrCompare, class H_Alpha>
static node_ptr insert_equal_upper_bound
(node_ptr h, node_ptr new_node, NodePtrCompare comp
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
std::size_t depth;
bstree_algo::insert_equal_upper_bound(h, new_node, comp, &depth);
rebalance_after_insertion(new_node, depth, tree_size+1, h_alpha, max_tree_size);
return new_node;
}

template<class NodePtrCompare, class H_Alpha>
static node_ptr insert_equal_lower_bound
(node_ptr h, node_ptr new_node, NodePtrCompare comp
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
std::size_t depth;
bstree_algo::insert_equal_lower_bound(h, new_node, comp, &depth);
rebalance_after_insertion(new_node, depth, tree_size+1, h_alpha, max_tree_size);
return new_node;
}

template<class NodePtrCompare, class H_Alpha>
static node_ptr insert_equal
(node_ptr header, node_ptr hint, node_ptr new_node, NodePtrCompare comp
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
std::size_t depth;
bstree_algo::insert_equal(header, hint, new_node, comp, &depth);
rebalance_after_insertion(new_node, depth, tree_size+1, h_alpha, max_tree_size);
return new_node;
}

template<class H_Alpha>
static node_ptr insert_before
(node_ptr header, node_ptr pos, node_ptr new_node
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
std::size_t depth;
bstree_algo::insert_before(header, pos, new_node, &depth);
rebalance_after_insertion(new_node, depth, tree_size+1, h_alpha, max_tree_size);
return new_node;
}

template<class H_Alpha>
static void push_back(node_ptr header, node_ptr new_node
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
std::size_t depth;
bstree_algo::push_back(header, new_node, &depth);
rebalance_after_insertion(new_node, depth, tree_size+1, h_alpha, max_tree_size);
}

template<class H_Alpha>
static void push_front(node_ptr header, node_ptr new_node
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
std::size_t depth;
bstree_algo::push_front(header, new_node, &depth);
rebalance_after_insertion(new_node, depth, tree_size+1, h_alpha, max_tree_size);
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, bool> insert_unique_check
(const_node_ptr header,  const KeyType &key
,KeyNodePtrCompare comp, insert_commit_data &commit_data)
{
std::size_t depth;
std::pair<node_ptr, bool> ret =
bstree_algo::insert_unique_check(header, key, comp, commit_data, &depth);
commit_data.depth = depth;
return ret;
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, bool> insert_unique_check
(const_node_ptr header, node_ptr hint, const KeyType &key
,KeyNodePtrCompare comp, insert_commit_data &commit_data)
{
std::size_t depth;
std::pair<node_ptr, bool> ret =
bstree_algo::insert_unique_check
(header, hint, key, comp, commit_data, &depth);
commit_data.depth = depth;
return ret;
}

template<class H_Alpha>
BOOST_INTRUSIVE_FORCEINLINE static void insert_unique_commit
(node_ptr header, node_ptr new_value, const insert_commit_data &commit_data
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{  return insert_commit(header, new_value, commit_data, tree_size, h_alpha, max_tree_size);  }

template<class NodePtrCompare, class H_Alpha, class AlphaByMaxSize>
static bool transfer_unique
( node_ptr header1, NodePtrCompare comp, std::size_t tree1_size, std::size_t &max_tree1_size
, node_ptr header2, node_ptr z,   std::size_t tree2_size, std::size_t &max_tree2_size
,H_Alpha h_alpha, AlphaByMaxSize alpha_by_maxsize)
{
insert_commit_data commit_data;
bool const transferable = insert_unique_check(header1, z, comp, commit_data).second;
if(transferable){
erase(header2, z, tree2_size, max_tree2_size, alpha_by_maxsize);
insert_commit(header1, z, commit_data, tree1_size, h_alpha, max_tree1_size);
}
return transferable;
}

template<class NodePtrCompare, class H_Alpha, class AlphaByMaxSize>
static void transfer_equal
( node_ptr header1, NodePtrCompare comp, std::size_t tree1_size, std::size_t &max_tree1_size
, node_ptr header2, node_ptr z,   std::size_t tree2_size, std::size_t &max_tree2_size
,H_Alpha h_alpha, AlphaByMaxSize alpha_by_maxsize)
{
insert_commit_data commit_data;
insert_equal_upper_bound_check(header1, z, comp, commit_data);
erase(header2, z, tree2_size, max_tree2_size, alpha_by_maxsize);
insert_commit(header1, z, commit_data, tree1_size, h_alpha, max_tree1_size);
}

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
static bool is_header(const_node_ptr p);

static void rebalance(node_ptr header);

static node_ptr rebalance_subtree(node_ptr old_root)
#endif   

private:

template<class KeyType, class KeyNodePtrCompare>
static void insert_equal_upper_bound_check
(node_ptr header,  const KeyType &key
,KeyNodePtrCompare comp, insert_commit_data &commit_data)
{
std::size_t depth;
bstree_algo::insert_equal_upper_bound_check(header, key, comp, commit_data, &depth);
commit_data.depth = depth;
}

template<class H_Alpha>
static void insert_commit
(node_ptr header, node_ptr new_value, const insert_commit_data &commit_data
,std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
bstree_algo::insert_unique_commit(header, new_value, commit_data);
rebalance_after_insertion(new_value, commit_data.depth, tree_size+1, h_alpha, max_tree_size);
}

template<class H_Alpha>
static void rebalance_after_insertion
(node_ptr x, std::size_t depth
, std::size_t tree_size, H_Alpha h_alpha, std::size_t &max_tree_size)
{
if(tree_size > max_tree_size)
max_tree_size = tree_size;

if(tree_size > 2 && 
depth > h_alpha(tree_size)){

node_ptr s = x;
std::size_t size = 1;
for(std::size_t ancestor = 1; ancestor != depth; ++ancestor){
const node_ptr s_parent = NodeTraits::get_parent(s);
const node_ptr s_parent_left = NodeTraits::get_left(s_parent);
const node_ptr s_sibling = s_parent_left == s ? NodeTraits::get_right(s_parent) : s_parent_left;
size += 1 + bstree_algo::subtree_size(s_sibling);
s = s_parent;
if(ancestor > h_alpha(size)){ 
bstree_algo::rebalance_subtree(s);
return;
}
}
max_tree_size = tree_size;
bstree_algo::rebalance_subtree(NodeTraits::get_parent(s));
}
}
};


template<class NodeTraits>
struct get_algo<SgTreeAlgorithms, NodeTraits>
{
typedef sgtree_algorithms<NodeTraits> type;
};

template <class ValueTraits, class NodePtrCompare, class ExtraChecker>
struct get_node_checker<SgTreeAlgorithms, ValueTraits, NodePtrCompare, ExtraChecker>
{
typedef detail::bstree_node_checker<ValueTraits, NodePtrCompare, ExtraChecker> type;
};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
