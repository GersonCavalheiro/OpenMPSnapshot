
#ifndef BOOST_INTRUSIVE_TREAP_ALGORITHMS_HPP
#define BOOST_INTRUSIVE_TREAP_ALGORITHMS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <cstddef>

#include <boost/intrusive/detail/assert.hpp>
#include <boost/intrusive/detail/algo_type.hpp>
#include <boost/intrusive/bstree_algorithms.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED

namespace detail
{

template<class ValueTraits, class NodePtrPrioCompare, class ExtraChecker>
struct treap_node_extra_checker
: public ExtraChecker
{
typedef ExtraChecker                            base_checker_t;
typedef ValueTraits                             value_traits;
typedef typename value_traits::node_traits      node_traits;
typedef typename node_traits::const_node_ptr const_node_ptr;

typedef typename base_checker_t::return_type    return_type;

treap_node_extra_checker(const NodePtrPrioCompare& prio_comp, ExtraChecker extra_checker)
: base_checker_t(extra_checker), prio_comp_(prio_comp)
{}

void operator () (const const_node_ptr& p,
const return_type& check_return_left, const return_type& check_return_right,
return_type& check_return)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT(!node_traits::get_left(p) || !prio_comp_(node_traits::get_left(p), p));
BOOST_INTRUSIVE_INVARIANT_ASSERT(!node_traits::get_right(p) || !prio_comp_(node_traits::get_right(p), p));
base_checker_t::operator()(p, check_return_left, check_return_right, check_return);
}

const NodePtrPrioCompare prio_comp_;
};

} 

#endif   

template<class NodeTraits>
class treap_algorithms
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
: public bstree_algorithms<NodeTraits>
#endif
{
public:
typedef NodeTraits                           node_traits;
typedef typename NodeTraits::node            node;
typedef typename NodeTraits::node_ptr       node_ptr;
typedef typename NodeTraits::const_node_ptr const_node_ptr;

private:

typedef bstree_algorithms<NodeTraits>  bstree_algo;

class rerotate_on_destroy
{
rerotate_on_destroy& operator=(const rerotate_on_destroy&);

public:
rerotate_on_destroy(node_ptr header, node_ptr p, std::size_t &n)
:  header_(header), p_(p), n_(n), remove_it_(true)
{}

~rerotate_on_destroy()
{
if(remove_it_){
rotate_up_n(header_, p_, n_);
}
}

void release()
{  remove_it_ = false;  }

const node_ptr header_;
const node_ptr p_;
std::size_t &n_;
bool remove_it_;
};

static void rotate_up_n(const node_ptr header, const node_ptr p, std::size_t n)
{
node_ptr p_parent(NodeTraits::get_parent(p));
node_ptr p_grandparent(NodeTraits::get_parent(p_parent));
while(n--){
if(p == NodeTraits::get_left(p_parent)){  
bstree_algo::rotate_right(p_parent, p, p_grandparent, header);
}
else{ 
bstree_algo::rotate_left(p_parent, p, p_grandparent, header);
}
p_parent      = p_grandparent;
p_grandparent = NodeTraits::get_parent(p_parent);
}
}


public:
struct insert_commit_data
:  public bstree_algo::insert_commit_data
{
std::size_t rotations;
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
#endif   

template<class NodePtrPriorityCompare>
static void unlink(node_ptr node, NodePtrPriorityCompare pcomp)
{
node_ptr x = NodeTraits::get_parent(node);
if(x){
while(!bstree_algo::is_header(x))
x = NodeTraits::get_parent(x);
erase(x, node, pcomp);
}
}

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
static node_ptr unlink_leftmost_without_rebalance(node_ptr header);

static bool unique(const_node_ptr node);

static std::size_t size(const_node_ptr header);

static node_ptr next_node(node_ptr node);

static node_ptr prev_node(node_ptr node);

static void init(node_ptr node);

static void init_header(node_ptr header);
#endif   

template<class NodePtrPriorityCompare>
static node_ptr erase(node_ptr header, node_ptr z, NodePtrPriorityCompare pcomp)
{
rebalance_for_erasure(header, z, pcomp);
bstree_algo::erase(header, z);
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

template<class NodePtrCompare, class NodePtrPriorityCompare>
static node_ptr insert_equal_upper_bound
(node_ptr h, node_ptr new_node, NodePtrCompare comp, NodePtrPriorityCompare pcomp)
{
insert_commit_data commit_data;
bstree_algo::insert_equal_upper_bound_check(h, new_node, comp, commit_data);
rebalance_check_and_commit(h, new_node, pcomp, commit_data);
return new_node;
}

template<class NodePtrCompare, class NodePtrPriorityCompare>
static node_ptr insert_equal_lower_bound
(node_ptr h, node_ptr new_node, NodePtrCompare comp, NodePtrPriorityCompare pcomp)
{
insert_commit_data commit_data;
bstree_algo::insert_equal_lower_bound_check(h, new_node, comp, commit_data);
rebalance_check_and_commit(h, new_node, pcomp, commit_data);
return new_node;
}

template<class NodePtrCompare, class NodePtrPriorityCompare>
static node_ptr insert_equal
(node_ptr h, node_ptr hint, node_ptr new_node, NodePtrCompare comp, NodePtrPriorityCompare pcomp)
{
insert_commit_data commit_data;
bstree_algo::insert_equal_check(h, hint, new_node, comp, commit_data);
rebalance_check_and_commit(h, new_node, pcomp, commit_data);
return new_node;
}

template<class NodePtrPriorityCompare>
static node_ptr insert_before
(node_ptr header, node_ptr pos, node_ptr new_node, NodePtrPriorityCompare pcomp)
{
insert_commit_data commit_data;
bstree_algo::insert_before_check(header, pos, commit_data);
rebalance_check_and_commit(header, new_node, pcomp, commit_data);
return new_node;
}

template<class NodePtrPriorityCompare>
static void push_back(node_ptr header, node_ptr new_node, NodePtrPriorityCompare pcomp)
{
insert_commit_data commit_data;
bstree_algo::push_back_check(header, commit_data);
rebalance_check_and_commit(header, new_node, pcomp, commit_data);
}

template<class NodePtrPriorityCompare>
static void push_front(node_ptr header, node_ptr new_node, NodePtrPriorityCompare pcomp)
{
insert_commit_data commit_data;
bstree_algo::push_front_check(header, commit_data);
rebalance_check_and_commit(header, new_node, pcomp, commit_data);
}

template<class KeyType, class KeyNodePtrCompare, class PrioType, class PrioNodePtrPrioCompare>
static std::pair<node_ptr, bool> insert_unique_check
( const_node_ptr header
, const KeyType &key, KeyNodePtrCompare comp
, const PrioType &prio, PrioNodePtrPrioCompare pcomp
, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> ret =
bstree_algo::insert_unique_check(header, key, comp, commit_data);
if(ret.second)
rebalance_after_insertion_check(header, commit_data.node, prio, pcomp, commit_data.rotations);
return ret;
}

template<class KeyType, class KeyNodePtrCompare, class PrioType, class PrioNodePtrPrioCompare>
static std::pair<node_ptr, bool> insert_unique_check
( const_node_ptr header, node_ptr hint
, const KeyType &key, KeyNodePtrCompare comp
, const PrioType &prio, PrioNodePtrPrioCompare pcomp
, insert_commit_data &commit_data)
{
std::pair<node_ptr, bool> ret =
bstree_algo::insert_unique_check(header, hint, key, comp, commit_data);
if(ret.second)
rebalance_after_insertion_check(header, commit_data.node, prio, pcomp, commit_data.rotations);
return ret;
}

static void insert_unique_commit
(node_ptr header, node_ptr new_node, const insert_commit_data &commit_data)
{
bstree_algo::insert_unique_commit(header, new_node, commit_data);
rotate_up_n(header, new_node, commit_data.rotations);
}

template<class NodePtrCompare, class PrioNodePtrPrioCompare>
static bool transfer_unique
(node_ptr header1, NodePtrCompare comp, PrioNodePtrPrioCompare pcomp, node_ptr header2, node_ptr z)
{
insert_commit_data commit_data;
bool const transferable = insert_unique_check(header1, z, comp, z, pcomp, commit_data).second;
if(transferable){
erase(header2, z, pcomp);
insert_unique_commit(header1, z, commit_data);         
}
return transferable;
}

template<class NodePtrCompare, class PrioNodePtrPrioCompare>
static void transfer_equal
(node_ptr header1, NodePtrCompare comp, PrioNodePtrPrioCompare pcomp, node_ptr header2, node_ptr z)
{
insert_commit_data commit_data;
bstree_algo::insert_equal_upper_bound_check(header1, z, comp, commit_data);
rebalance_after_insertion_check(header1, commit_data.node, z, pcomp, commit_data.rotations);
rebalance_for_erasure(header2, z, pcomp);
bstree_algo::erase(header2, z);
bstree_algo::insert_unique_commit(header1, z, commit_data);
rotate_up_n(header1, z, commit_data.rotations);
}

#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED

static bool is_header(const_node_ptr p);
#endif   

private:

template<class NodePtrPriorityCompare>
static void rebalance_for_erasure(node_ptr header, node_ptr z, NodePtrPriorityCompare pcomp)
{
std::size_t n = 0;
rerotate_on_destroy rb(header, z, n);

node_ptr z_left  = NodeTraits::get_left(z);
node_ptr z_right = NodeTraits::get_right(z);
while(z_left || z_right){
const node_ptr z_parent(NodeTraits::get_parent(z));
if(!z_right || (z_left && pcomp(z_left, z_right))){
bstree_algo::rotate_right(z, z_left, z_parent, header);
}
else{
bstree_algo::rotate_left(z, z_right, z_parent, header);
}
++n;
z_left  = NodeTraits::get_left(z);
z_right = NodeTraits::get_right(z);
}
rb.release();
}

template<class NodePtrPriorityCompare>
static void rebalance_check_and_commit
(node_ptr h, node_ptr new_node, NodePtrPriorityCompare pcomp, insert_commit_data &commit_data)
{
rebalance_after_insertion_check(h, commit_data.node, new_node, pcomp, commit_data.rotations);
bstree_algo::insert_unique_commit(h, new_node, commit_data);
rotate_up_n(h, new_node, commit_data.rotations);
}

template<class Key, class KeyNodePriorityCompare>
static void rebalance_after_insertion_check
(const_node_ptr header, const_node_ptr up, const Key &k
, KeyNodePriorityCompare pcomp, std::size_t &num_rotations)
{
const_node_ptr upnode(up);
num_rotations = 0;
std::size_t n = 0;
while(upnode != header && pcomp(k, upnode)){
++n;
upnode = NodeTraits::get_parent(upnode);
}
num_rotations = n;
}

template<class NodePtrPriorityCompare>
static bool check_invariant(const_node_ptr header, NodePtrPriorityCompare pcomp)
{
node_ptr beg = begin_node(header);
node_ptr end = end_node(header);

while(beg != end){
node_ptr p = NodeTraits::get_parent(beg);
if(p != header){
if(pcomp(beg, p))
return false;
}
beg = next_node(beg);
}
return true;
}

};


template<class NodeTraits>
struct get_algo<TreapAlgorithms, NodeTraits>
{
typedef treap_algorithms<NodeTraits> type;
};

template <class ValueTraits, class NodePtrCompare, class ExtraChecker>
struct get_node_checker<TreapAlgorithms, ValueTraits, NodePtrCompare, ExtraChecker>
{
typedef detail::bstree_node_checker<ValueTraits, NodePtrCompare, ExtraChecker> type;
};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
