
#ifndef BOOST_INTRUSIVE_BSTREE_ALGORITHMS_HPP
#define BOOST_INTRUSIVE_BSTREE_ALGORITHMS_HPP

#include <cstddef>
#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/bstree_algorithms_base.hpp>
#include <boost/intrusive/detail/assert.hpp>
#include <boost/intrusive/detail/uncast.hpp>
#include <boost/intrusive/detail/math.hpp>
#include <boost/intrusive/detail/algo_type.hpp>

#include <boost/intrusive/detail/minimal_pair_header.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


template <class NodePtr>
struct insert_commit_data_t
{
BOOST_INTRUSIVE_FORCEINLINE insert_commit_data_t()
: link_left(false), node()
{}
bool     link_left;
NodePtr  node;
};

template <class NodePtr>
struct data_for_rebalance_t
{
NodePtr  x;
NodePtr  x_parent;
NodePtr  y;
};

namespace detail {

template<class ValueTraits, class NodePtrCompare, class ExtraChecker>
struct bstree_node_checker
: public ExtraChecker
{
typedef ExtraChecker                            base_checker_t;
typedef ValueTraits                             value_traits;
typedef typename value_traits::node_traits      node_traits;
typedef typename node_traits::const_node_ptr    const_node_ptr;

struct return_type
: public base_checker_t::return_type
{
BOOST_INTRUSIVE_FORCEINLINE return_type()
: min_key_node_ptr(const_node_ptr()), max_key_node_ptr(const_node_ptr()), node_count(0)
{}

const_node_ptr min_key_node_ptr;
const_node_ptr max_key_node_ptr;
size_t   node_count;
};

BOOST_INTRUSIVE_FORCEINLINE bstree_node_checker(const NodePtrCompare& comp, ExtraChecker extra_checker)
: base_checker_t(extra_checker), comp_(comp)
{}

void operator () (const const_node_ptr& p,
const return_type& check_return_left, const return_type& check_return_right,
return_type& check_return)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT(!check_return_left.max_key_node_ptr || !comp_(p, check_return_left.max_key_node_ptr));
BOOST_INTRUSIVE_INVARIANT_ASSERT(!check_return_right.min_key_node_ptr || !comp_(check_return_right.min_key_node_ptr, p));
check_return.min_key_node_ptr = node_traits::get_left(p)? check_return_left.min_key_node_ptr : p;
check_return.max_key_node_ptr = node_traits::get_right(p)? check_return_right.max_key_node_ptr : p;
check_return.node_count = check_return_left.node_count + check_return_right.node_count + 1;
base_checker_t::operator()(p, check_return_left, check_return_right, check_return);
}

const NodePtrCompare comp_;
};

} 




template<class NodeTraits>
class bstree_algorithms : public bstree_algorithms_base<NodeTraits>
{
public:
typedef typename NodeTraits::node            node;
typedef NodeTraits                           node_traits;
typedef typename NodeTraits::node_ptr        node_ptr;
typedef typename NodeTraits::const_node_ptr  const_node_ptr;
typedef insert_commit_data_t<node_ptr>       insert_commit_data;
typedef data_for_rebalance_t<node_ptr>       data_for_rebalance;

typedef bstree_algorithms<NodeTraits>        this_type;
typedef bstree_algorithms_base<NodeTraits>   base_type;
private:
template<class Disposer>
struct dispose_subtree_disposer
{
BOOST_INTRUSIVE_FORCEINLINE dispose_subtree_disposer(Disposer &disp, const node_ptr & subtree)
: disposer_(&disp), subtree_(subtree)
{}

BOOST_INTRUSIVE_FORCEINLINE void release()
{  disposer_ = 0;  }

BOOST_INTRUSIVE_FORCEINLINE ~dispose_subtree_disposer()
{
if(disposer_){
dispose_subtree(subtree_, *disposer_);
}
}
Disposer *disposer_;
const node_ptr subtree_;
};


public:
BOOST_INTRUSIVE_FORCEINLINE static node_ptr begin_node(const const_node_ptr & header)
{  return node_traits::get_left(header);   }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr end_node(const const_node_ptr & header)
{  return detail::uncast(header);   }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr root_node(const const_node_ptr & header)
{
node_ptr p = node_traits::get_parent(header);
return p ? p : detail::uncast(header);
}

BOOST_INTRUSIVE_FORCEINLINE static bool unique(const const_node_ptr & node)
{ return !NodeTraits::get_parent(node); }

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
static node_ptr get_header(const const_node_ptr & node);
#endif

static void swap_nodes(node_ptr node1, node_ptr node2)
{
if(node1 == node2)
return;

node_ptr header1(base_type::get_header(node1)), header2(base_type::get_header(node2));
swap_nodes(node1, header1, node2, header2);
}

static void swap_nodes(node_ptr node1, node_ptr header1, node_ptr node2, node_ptr header2)
{
if(node1 == node2)
return;

if(header1 != header2){
if(node1 == NodeTraits::get_left(header1)){
NodeTraits::set_left(header1, node2);
}

if(node1 == NodeTraits::get_right(header1)){
NodeTraits::set_right(header1, node2);
}

if(node1 == NodeTraits::get_parent(header1)){
NodeTraits::set_parent(header1, node2);
}

if(node2 == NodeTraits::get_left(header2)){
NodeTraits::set_left(header2, node1);
}

if(node2 == NodeTraits::get_right(header2)){
NodeTraits::set_right(header2, node1);
}

if(node2 == NodeTraits::get_parent(header2)){
NodeTraits::set_parent(header2, node1);
}
}
else{
if(node1 == NodeTraits::get_left(header1)){
NodeTraits::set_left(header1, node2);
}
else if(node2 == NodeTraits::get_left(header2)){
NodeTraits::set_left(header2, node1);
}

if(node1 == NodeTraits::get_right(header1)){
NodeTraits::set_right(header1, node2);
}
else if(node2 == NodeTraits::get_right(header2)){
NodeTraits::set_right(header2, node1);
}

if(node1 == NodeTraits::get_parent(header1)){
NodeTraits::set_parent(header1, node2);
}
else if(node2 == NodeTraits::get_parent(header2)){
NodeTraits::set_parent(header2, node1);
}

if(node1 == NodeTraits::get_parent(node2)){
NodeTraits::set_parent(node2, node2);

if(node2 == NodeTraits::get_right(node1)){
NodeTraits::set_right(node1, node1);
}
else{
NodeTraits::set_left(node1, node1);
}
}
else if(node2 == NodeTraits::get_parent(node1)){
NodeTraits::set_parent(node1, node1);

if(node1 == NodeTraits::get_right(node2)){
NodeTraits::set_right(node2, node2);
}
else{
NodeTraits::set_left(node2, node2);
}
}
}

node_ptr temp;
temp = NodeTraits::get_left(node1);
NodeTraits::set_left(node1, NodeTraits::get_left(node2));
NodeTraits::set_left(node2, temp);
temp = NodeTraits::get_right(node1);
NodeTraits::set_right(node1, NodeTraits::get_right(node2));
NodeTraits::set_right(node2, temp);
temp = NodeTraits::get_parent(node1);
NodeTraits::set_parent(node1, NodeTraits::get_parent(node2));
NodeTraits::set_parent(node2, temp);

if((temp = NodeTraits::get_left(node1))){
NodeTraits::set_parent(temp, node1);
}
if((temp = NodeTraits::get_right(node1))){
NodeTraits::set_parent(temp, node1);
}
if((temp = NodeTraits::get_parent(node1)) &&
temp != header2){
if(NodeTraits::get_left(temp) == node2){
NodeTraits::set_left(temp, node1);
}
if(NodeTraits::get_right(temp) == node2){
NodeTraits::set_right(temp, node1);
}
}
if((temp = NodeTraits::get_left(node2))){
NodeTraits::set_parent(temp, node2);
}
if((temp = NodeTraits::get_right(node2))){
NodeTraits::set_parent(temp, node2);
}
if((temp = NodeTraits::get_parent(node2)) &&
temp != header1){
if(NodeTraits::get_left(temp) == node1){
NodeTraits::set_left(temp, node2);
}
if(NodeTraits::get_right(temp) == node1){
NodeTraits::set_right(temp, node2);
}
}
}

BOOST_INTRUSIVE_FORCEINLINE static void replace_node(node_ptr node_to_be_replaced, node_ptr new_node)
{
if(node_to_be_replaced == new_node)
return;
replace_node(node_to_be_replaced, base_type::get_header(node_to_be_replaced), new_node);
}

static void replace_node(node_ptr node_to_be_replaced, node_ptr header, node_ptr new_node)
{
if(node_to_be_replaced == new_node)
return;

if(node_to_be_replaced == NodeTraits::get_left(header)){
NodeTraits::set_left(header, new_node);
}

if(node_to_be_replaced == NodeTraits::get_right(header)){
NodeTraits::set_right(header, new_node);
}

if(node_to_be_replaced == NodeTraits::get_parent(header)){
NodeTraits::set_parent(header, new_node);
}

node_ptr temp;
NodeTraits::set_left(new_node, NodeTraits::get_left(node_to_be_replaced));
NodeTraits::set_right(new_node, NodeTraits::get_right(node_to_be_replaced));
NodeTraits::set_parent(new_node, NodeTraits::get_parent(node_to_be_replaced));

if((temp = NodeTraits::get_left(new_node))){
NodeTraits::set_parent(temp, new_node);
}
if((temp = NodeTraits::get_right(new_node))){
NodeTraits::set_parent(temp, new_node);
}
if((temp = NodeTraits::get_parent(new_node)) &&
temp != header){
if(NodeTraits::get_left(temp) == node_to_be_replaced){
NodeTraits::set_left(temp, new_node);
}
if(NodeTraits::get_right(temp) == node_to_be_replaced){
NodeTraits::set_right(temp, new_node);
}
}
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
static node_ptr next_node(const node_ptr & node);

static node_ptr prev_node(const node_ptr & node);

static node_ptr minimum(node_ptr node);

static node_ptr maximum(node_ptr node);
#endif

BOOST_INTRUSIVE_FORCEINLINE static void init(node_ptr node)
{
NodeTraits::set_parent(node, node_ptr());
NodeTraits::set_left(node, node_ptr());
NodeTraits::set_right(node, node_ptr());
}

BOOST_INTRUSIVE_FORCEINLINE static bool inited(const const_node_ptr & node)
{
return !NodeTraits::get_parent(node) &&
!NodeTraits::get_left(node)   &&
!NodeTraits::get_right(node)  ;
}

BOOST_INTRUSIVE_FORCEINLINE static void init_header(node_ptr header)
{
NodeTraits::set_parent(header, node_ptr());
NodeTraits::set_left(header, header);
NodeTraits::set_right(header, header);
}

template<class Disposer>
static void clear_and_dispose(const node_ptr & header, Disposer disposer)
{
node_ptr source_root = NodeTraits::get_parent(header);
if(!source_root)
return;
dispose_subtree(source_root, disposer);
init_header(header);
}

static node_ptr unlink_leftmost_without_rebalance(node_ptr header)
{
node_ptr leftmost = NodeTraits::get_left(header);
if (leftmost == header)
return node_ptr();
node_ptr leftmost_parent(NodeTraits::get_parent(leftmost));
node_ptr leftmost_right (NodeTraits::get_right(leftmost));
bool is_root = leftmost_parent == header;

if (leftmost_right){
NodeTraits::set_parent(leftmost_right, leftmost_parent);
NodeTraits::set_left(header, base_type::minimum(leftmost_right));

if (is_root)
NodeTraits::set_parent(header, leftmost_right);
else
NodeTraits::set_left(NodeTraits::get_parent(header), leftmost_right);
}
else if (is_root){
NodeTraits::set_parent(header, node_ptr());
NodeTraits::set_left(header,  header);
NodeTraits::set_right(header, header);
}
else{
NodeTraits::set_left(leftmost_parent, node_ptr());
NodeTraits::set_left(header, leftmost_parent);
}
return leftmost;
}

static std::size_t size(const const_node_ptr & header)
{
node_ptr beg(begin_node(header));
node_ptr end(end_node(header));
std::size_t i = 0;
for(;beg != end; beg = base_type::next_node(beg)) ++i;
return i;
}

static void swap_tree(node_ptr header1, node_ptr header2)
{
if(header1 == header2)
return;

node_ptr tmp;

tmp = NodeTraits::get_parent(header1);
NodeTraits::set_parent(header1, NodeTraits::get_parent(header2));
NodeTraits::set_parent(header2, tmp);
tmp = NodeTraits::get_left(header1);
NodeTraits::set_left(header1, NodeTraits::get_left(header2));
NodeTraits::set_left(header2, tmp);
tmp = NodeTraits::get_right(header1);
NodeTraits::set_right(header1, NodeTraits::get_right(header2));
NodeTraits::set_right(header2, tmp);

node_ptr h1_parent(NodeTraits::get_parent(header1));
if(h1_parent){
NodeTraits::set_parent(h1_parent, header1);
}
else{
NodeTraits::set_left(header1, header1);
NodeTraits::set_right(header1, header1);
}

node_ptr h2_parent(NodeTraits::get_parent(header2));
if(h2_parent){
NodeTraits::set_parent(h2_parent, header2);
}
else{
NodeTraits::set_left(header2, header2);
NodeTraits::set_right(header2, header2);
}
}

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)
static bool is_header(const const_node_ptr & p);
#endif

template<class KeyType, class KeyNodePtrCompare>
static node_ptr find
(const const_node_ptr & header, const KeyType &key, KeyNodePtrCompare comp)
{
node_ptr end = detail::uncast(header);
node_ptr y = lower_bound(header, key, comp);
return (y == end || comp(key, y)) ? end : y;
}

template< class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> bounded_range
( const const_node_ptr & header
, const KeyType &lower_key
, const KeyType &upper_key
, KeyNodePtrCompare comp
, bool left_closed
, bool right_closed)
{
node_ptr y = detail::uncast(header);
node_ptr x = NodeTraits::get_parent(header);

while(x){
if(comp(x, lower_key)){
BOOST_INTRUSIVE_INVARIANT_ASSERT(comp(x, upper_key));
x = NodeTraits::get_right(x);
}
else if(comp(upper_key, x)){
y = x;
x = NodeTraits::get_left(x);
}
else{
BOOST_INTRUSIVE_INVARIANT_ASSERT(left_closed || right_closed || comp(lower_key, x) || comp(x, upper_key));
return std::pair<node_ptr,node_ptr>(
left_closed
? lower_bound_loop(NodeTraits::get_left(x), x, lower_key, comp)
: upper_bound_loop(x, y, lower_key, comp)
,
right_closed
? upper_bound_loop(NodeTraits::get_right(x), y, upper_key, comp)
: lower_bound_loop(x, y, upper_key, comp)
);
}
}
return std::pair<node_ptr,node_ptr> (y, y);
}

template<class KeyType, class KeyNodePtrCompare>
static std::size_t count
(const const_node_ptr & header, const KeyType &key, KeyNodePtrCompare comp)
{
std::pair<node_ptr, node_ptr> ret = equal_range(header, key, comp);
std::size_t n = 0;
while(ret.first != ret.second){
++n;
ret.first = base_type::next_node(ret.first);
}
return n;
}

template<class KeyType, class KeyNodePtrCompare>
BOOST_INTRUSIVE_FORCEINLINE static std::pair<node_ptr, node_ptr> equal_range
(const const_node_ptr & header, const KeyType &key, KeyNodePtrCompare comp)
{
return bounded_range(header, key, key, comp, true, true);
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, node_ptr> lower_bound_range
(const const_node_ptr & header, const KeyType &key, KeyNodePtrCompare comp)
{
node_ptr const lb(lower_bound(header, key, comp));
std::pair<node_ptr, node_ptr> ret_ii(lb, lb);
if(lb != header && !comp(key, lb)){
ret_ii.second = base_type::next_node(ret_ii.second);
}
return ret_ii;
}

template<class KeyType, class KeyNodePtrCompare>
BOOST_INTRUSIVE_FORCEINLINE static node_ptr lower_bound
(const const_node_ptr & header, const KeyType &key, KeyNodePtrCompare comp)
{
return lower_bound_loop(NodeTraits::get_parent(header), detail::uncast(header), key, comp);
}

template<class KeyType, class KeyNodePtrCompare>
BOOST_INTRUSIVE_FORCEINLINE static node_ptr upper_bound
(const const_node_ptr & header, const KeyType &key, KeyNodePtrCompare comp)
{
return upper_bound_loop(NodeTraits::get_parent(header), detail::uncast(header), key, comp);
}

BOOST_INTRUSIVE_FORCEINLINE static void insert_unique_commit
(node_ptr header, node_ptr new_value, const insert_commit_data &commit_data)
{  return insert_commit(header, new_value, commit_data); }

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, bool> insert_unique_check
(const const_node_ptr & header, const KeyType &key
,KeyNodePtrCompare comp, insert_commit_data &commit_data
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
std::size_t depth = 0;
node_ptr h(detail::uncast(header));
node_ptr y(h);
node_ptr x(NodeTraits::get_parent(y));
node_ptr prev = node_ptr();

bool left_child = true;
while(x){
++depth;
y = x;
left_child = comp(key, x);
x = left_child ?
NodeTraits::get_left(x) : (prev = y, NodeTraits::get_right(x));
}

if(pdepth)  *pdepth = depth;

const bool not_present = !prev || comp(prev, key);
if(not_present){
commit_data.link_left = left_child;
commit_data.node      = y;
}
return std::pair<node_ptr, bool>(prev, not_present);
}

template<class KeyType, class KeyNodePtrCompare>
static std::pair<node_ptr, bool> insert_unique_check
(const const_node_ptr & header, const node_ptr &hint, const KeyType &key
,KeyNodePtrCompare comp, insert_commit_data &commit_data
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
if(hint == header || comp(key, hint)){
node_ptr prev(hint);
if(hint == begin_node(header) || comp((prev = base_type::prev_node(hint)), key)){
commit_data.link_left = unique(header) || !NodeTraits::get_left(hint);
commit_data.node      = commit_data.link_left ? hint : prev;
if(pdepth){
*pdepth = commit_data.node == header ? 0 : depth(commit_data.node) + 1;
}
return std::pair<node_ptr, bool>(node_ptr(), true);
}
}
return insert_unique_check(header, key, comp, commit_data, pdepth);
}

template<class NodePtrCompare>
static node_ptr insert_equal
(node_ptr h, node_ptr hint, node_ptr new_node, NodePtrCompare comp
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
insert_commit_data commit_data;
insert_equal_check(h, hint, new_node, comp, commit_data, pdepth);
insert_commit(h, new_node, commit_data);
return new_node;
}

template<class NodePtrCompare>
static node_ptr insert_equal_upper_bound
(node_ptr h, node_ptr new_node, NodePtrCompare comp
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
insert_commit_data commit_data;
insert_equal_upper_bound_check(h, new_node, comp, commit_data, pdepth);
insert_commit(h, new_node, commit_data);
return new_node;
}

template<class NodePtrCompare>
static node_ptr insert_equal_lower_bound
(node_ptr h, node_ptr new_node, NodePtrCompare comp
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
insert_commit_data commit_data;
insert_equal_lower_bound_check(h, new_node, comp, commit_data, pdepth);
insert_commit(h, new_node, commit_data);
return new_node;
}

static node_ptr insert_before
(node_ptr header, node_ptr pos, node_ptr new_node
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
insert_commit_data commit_data;
insert_before_check(header, pos, commit_data, pdepth);
insert_commit(header, new_node, commit_data);
return new_node;
}

static void push_back
(node_ptr header, node_ptr new_node
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
insert_commit_data commit_data;
push_back_check(header, commit_data, pdepth);
insert_commit(header, new_node, commit_data);
}

static void push_front
(node_ptr header, node_ptr new_node
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
insert_commit_data commit_data;
push_front_check(header, commit_data, pdepth);
insert_commit(header, new_node, commit_data);
}

static std::size_t depth(const_node_ptr node)
{
std::size_t depth = 0;
node_ptr p_parent;
while(node != NodeTraits::get_parent(p_parent = NodeTraits::get_parent(node))){
++depth;
node = p_parent;
}
return depth;
}

template <class Cloner, class Disposer>
static void clone
(const const_node_ptr & source_header, node_ptr target_header, Cloner cloner, Disposer disposer)
{
if(!unique(target_header)){
clear_and_dispose(target_header, disposer);
}

node_ptr leftmost, rightmost;
node_ptr new_root = clone_subtree
(source_header, target_header, cloner, disposer, leftmost, rightmost);

NodeTraits::set_parent(target_header, new_root);
NodeTraits::set_left  (target_header, leftmost);
NodeTraits::set_right (target_header, rightmost);
}

BOOST_INTRUSIVE_FORCEINLINE static void erase(node_ptr header, node_ptr z)
{
data_for_rebalance ignored;
erase(header, z, ignored);
}

template<class NodePtrCompare>
BOOST_INTRUSIVE_FORCEINLINE static bool transfer_unique
(node_ptr header1, NodePtrCompare comp, node_ptr header2, node_ptr z)
{
data_for_rebalance ignored;
return transfer_unique(header1, comp, header2, z, ignored);
}

template<class NodePtrCompare>
BOOST_INTRUSIVE_FORCEINLINE static void transfer_equal
(node_ptr header1, NodePtrCompare comp, node_ptr header2, node_ptr z)
{
data_for_rebalance ignored;
transfer_equal(header1, comp, header2, z, ignored);
}

static void unlink(node_ptr node)
{
node_ptr x = NodeTraits::get_parent(node);
if(x){
while(!base_type::is_header(x))
x = NodeTraits::get_parent(x);
erase(x, node);
}
}

static void rebalance(node_ptr header)
{
node_ptr root = NodeTraits::get_parent(header);
if(root){
rebalance_subtree(root);
}
}

static node_ptr rebalance_subtree(node_ptr old_root)
{

node_ptr super_root = NodeTraits::get_parent(old_root);
BOOST_INTRUSIVE_INVARIANT_ASSERT(super_root);

node_ptr super_root_right_backup = NodeTraits::get_right(super_root);
bool super_root_is_header = NodeTraits::get_parent(super_root) == old_root;
bool old_root_is_right  = is_right_child(old_root);
NodeTraits::set_right(super_root, old_root);

std::size_t size;
subtree_to_vine(super_root, size);
vine_to_subtree(super_root, size);
node_ptr new_root = NodeTraits::get_right(super_root);

if(super_root_is_header){
NodeTraits::set_right(super_root, super_root_right_backup);
NodeTraits::set_parent(super_root, new_root);
}
else if(old_root_is_right){
NodeTraits::set_right(super_root, new_root);
}
else{
NodeTraits::set_right(super_root, super_root_right_backup);
NodeTraits::set_left(super_root, new_root);
}
return new_root;
}

template<class Checker>
static void check(const const_node_ptr& header, Checker checker, typename Checker::return_type& checker_return)
{
const_node_ptr root_node_ptr = NodeTraits::get_parent(header);
if (!root_node_ptr){
BOOST_INTRUSIVE_INVARIANT_ASSERT(NodeTraits::get_left(header) == header);
BOOST_INTRUSIVE_INVARIANT_ASSERT(NodeTraits::get_right(header) == header);
}
else{
BOOST_INTRUSIVE_INVARIANT_ASSERT(NodeTraits::get_parent(root_node_ptr) == header);
check_subtree(root_node_ptr, checker, checker_return);
const_node_ptr p = root_node_ptr;
while (NodeTraits::get_left(p)) { p = NodeTraits::get_left(p); }
BOOST_INTRUSIVE_INVARIANT_ASSERT(NodeTraits::get_left(header) == p);
p = root_node_ptr;
while (NodeTraits::get_right(p)) { p = NodeTraits::get_right(p); }
BOOST_INTRUSIVE_INVARIANT_ASSERT(NodeTraits::get_right(header) == p);
}
}

protected:

template<class NodePtrCompare>
static bool transfer_unique
(node_ptr header1, NodePtrCompare comp, node_ptr header2, node_ptr z, data_for_rebalance &info)
{
insert_commit_data commit_data;
bool const transferable = insert_unique_check(header1, z, comp, commit_data).second;
if(transferable){
erase(header2, z, info);
insert_commit(header1, z, commit_data);
}
return transferable;
}

template<class NodePtrCompare>
static void transfer_equal
(node_ptr header1, NodePtrCompare comp, node_ptr header2, node_ptr z, data_for_rebalance &info)
{
insert_commit_data commit_data;
insert_equal_upper_bound_check(header1, z, comp, commit_data);
erase(header2, z, info);
insert_commit(header1, z, commit_data);
}

static void erase(node_ptr header, node_ptr z, data_for_rebalance &info)
{
node_ptr y(z);
node_ptr x;
const node_ptr z_left(NodeTraits::get_left(z));
const node_ptr z_right(NodeTraits::get_right(z));

if(!z_left){
x = z_right;    
}
else if(!z_right){ 
x = z_left;       
BOOST_ASSERT(x);
}
else{ 
y = base_type::minimum(z_right);
x = NodeTraits::get_right(y);     
}

node_ptr x_parent;
const node_ptr z_parent(NodeTraits::get_parent(z));
const bool z_is_leftchild(NodeTraits::get_left(z_parent) == z);

if(y != z){ 
NodeTraits::set_parent(z_left, y);
NodeTraits::set_left(y, z_left);
if(y != z_right){
NodeTraits::set_right(y, z_right);
NodeTraits::set_parent(z_right, y);
x_parent = NodeTraits::get_parent(y);
BOOST_ASSERT(NodeTraits::get_left(x_parent) == y);
if(x)
NodeTraits::set_parent(x, x_parent);
NodeTraits::set_left(x_parent, x);
}
else{ 
x_parent = y;
}
NodeTraits::set_parent(y, z_parent);
this_type::set_child(header, y, z_parent, z_is_leftchild);
}
else {  
x_parent = z_parent;
if(x)
NodeTraits::set_parent(x, z_parent);
this_type::set_child(header, x, z_parent, z_is_leftchild);

if(NodeTraits::get_left(header) == z){
BOOST_ASSERT(!z_left);
NodeTraits::set_left(header, !z_right ?
z_parent :  
base_type::minimum(z_right));
}
if(NodeTraits::get_right(header) == z){
BOOST_ASSERT(!z_right);
NodeTraits::set_right(header, !z_left ?
z_parent :  
base_type::maximum(z_left));
}
}

info.x = x;
info.y = y;
BOOST_ASSERT(!x || NodeTraits::get_parent(x) == x_parent);
info.x_parent = x_parent;
}

static std::size_t subtree_size(const const_node_ptr & subtree)
{
std::size_t count = 0;
if (subtree){
node_ptr n = detail::uncast(subtree);
node_ptr m = NodeTraits::get_left(n);
while(m){
n = m;
m = NodeTraits::get_left(n);
}

while(1){
++count;
node_ptr n_right(NodeTraits::get_right(n));
if(n_right){
n = n_right;
m = NodeTraits::get_left(n);
while(m){
n = m;
m = NodeTraits::get_left(n);
}
}
else {
do{
if (n == subtree){
return count;
}
m = n;
n = NodeTraits::get_parent(n);
}while(NodeTraits::get_left(n) != m);
}
}
}
return count;
}

BOOST_INTRUSIVE_FORCEINLINE static bool is_left_child(const node_ptr & p)
{  return NodeTraits::get_left(NodeTraits::get_parent(p)) == p;  }

BOOST_INTRUSIVE_FORCEINLINE static bool is_right_child(const node_ptr & p)
{  return NodeTraits::get_right(NodeTraits::get_parent(p)) == p;  }

static void insert_before_check
(node_ptr header, node_ptr pos
, insert_commit_data &commit_data
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
node_ptr prev(pos);
if(pos != NodeTraits::get_left(header))
prev = base_type::prev_node(pos);
bool link_left = unique(header) || !NodeTraits::get_left(pos);
commit_data.link_left = link_left;
commit_data.node = link_left ? pos : prev;
if(pdepth){
*pdepth = commit_data.node == header ? 0 : depth(commit_data.node) + 1;
}
}

static void push_back_check
(node_ptr header, insert_commit_data &commit_data
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
node_ptr prev(NodeTraits::get_right(header));
if(pdepth){
*pdepth = prev == header ? 0 : depth(prev) + 1;
}
commit_data.link_left = false;
commit_data.node = prev;
}

static void push_front_check
(node_ptr header, insert_commit_data &commit_data
#ifndef BOOST_INTRUSIVE_DOXYGEN_INVOKED
, std::size_t *pdepth = 0
#endif
)
{
node_ptr pos(NodeTraits::get_left(header));
if(pdepth){
*pdepth = pos == header ? 0 : depth(pos) + 1;
}
commit_data.link_left = true;
commit_data.node = pos;
}

template<class NodePtrCompare>
static void insert_equal_check
(node_ptr header, node_ptr hint, node_ptr new_node, NodePtrCompare comp
, insert_commit_data &commit_data
, std::size_t *pdepth = 0
)
{
if(hint == header || !comp(hint, new_node)){
node_ptr prev(hint);
if(hint == NodeTraits::get_left(header) ||
!comp(new_node, (prev = base_type::prev_node(hint)))){
bool link_left = unique(header) || !NodeTraits::get_left(hint);
commit_data.link_left = link_left;
commit_data.node = link_left ? hint : prev;
if(pdepth){
*pdepth = commit_data.node == header ? 0 : depth(commit_data.node) + 1;
}
}
else{
insert_equal_upper_bound_check(header, new_node, comp, commit_data, pdepth);
}
}
else{
insert_equal_lower_bound_check(header, new_node, comp, commit_data, pdepth);
}
}

template<class NodePtrCompare>
static void insert_equal_upper_bound_check
(node_ptr h, node_ptr new_node, NodePtrCompare comp, insert_commit_data & commit_data, std::size_t *pdepth = 0)
{
std::size_t depth = 0;
node_ptr y(h);
node_ptr x(NodeTraits::get_parent(y));

while(x){
++depth;
y = x;
x = comp(new_node, x) ?
NodeTraits::get_left(x) : NodeTraits::get_right(x);
}
if(pdepth)  *pdepth = depth;
commit_data.link_left = (y == h) || comp(new_node, y);
commit_data.node = y;
}

template<class NodePtrCompare>
static void insert_equal_lower_bound_check
(node_ptr h, node_ptr new_node, NodePtrCompare comp, insert_commit_data & commit_data, std::size_t *pdepth = 0)
{
std::size_t depth = 0;
node_ptr y(h);
node_ptr x(NodeTraits::get_parent(y));

while(x){
++depth;
y = x;
x = !comp(x, new_node) ?
NodeTraits::get_left(x) : NodeTraits::get_right(x);
}
if(pdepth)  *pdepth = depth;
commit_data.link_left = (y == h) || !comp(y, new_node);
commit_data.node = y;
}

static void insert_commit
(node_ptr header, node_ptr new_node, const insert_commit_data &commit_data)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT(commit_data.node != node_ptr());
node_ptr parent_node(commit_data.node);
if(parent_node == header){
NodeTraits::set_parent(header, new_node);
NodeTraits::set_right(header, new_node);
NodeTraits::set_left(header, new_node);
}
else if(commit_data.link_left){
NodeTraits::set_left(parent_node, new_node);
if(parent_node == NodeTraits::get_left(header))
NodeTraits::set_left(header, new_node);
}
else{
NodeTraits::set_right(parent_node, new_node);
if(parent_node == NodeTraits::get_right(header))
NodeTraits::set_right(header, new_node);
}
NodeTraits::set_parent(new_node, parent_node);
NodeTraits::set_right(new_node, node_ptr());
NodeTraits::set_left(new_node, node_ptr());
}

static void set_child(node_ptr header, node_ptr new_child, node_ptr new_parent, const bool link_left)
{
if(new_parent == header)
NodeTraits::set_parent(header, new_child);
else if(link_left)
NodeTraits::set_left(new_parent, new_child);
else
NodeTraits::set_right(new_parent, new_child);
}

static void rotate_left_no_parent_fix(node_ptr p, node_ptr p_right)
{
node_ptr p_right_left(NodeTraits::get_left(p_right));
NodeTraits::set_right(p, p_right_left);
if(p_right_left){
NodeTraits::set_parent(p_right_left, p);
}
NodeTraits::set_left(p_right, p);
NodeTraits::set_parent(p, p_right);
}

static void rotate_left(node_ptr p, node_ptr p_right, node_ptr p_parent, node_ptr header)
{
const bool p_was_left(NodeTraits::get_left(p_parent) == p);
rotate_left_no_parent_fix(p, p_right);
NodeTraits::set_parent(p_right, p_parent);
set_child(header, p_right, p_parent, p_was_left);
}

static void rotate_right_no_parent_fix(node_ptr p, node_ptr p_left)
{
node_ptr p_left_right(NodeTraits::get_right(p_left));
NodeTraits::set_left(p, p_left_right);
if(p_left_right){
NodeTraits::set_parent(p_left_right, p);
}
NodeTraits::set_right(p_left, p);
NodeTraits::set_parent(p, p_left);
}

static void rotate_right(node_ptr p, node_ptr p_left, node_ptr p_parent, node_ptr header)
{
const bool p_was_left(NodeTraits::get_left(p_parent) == p);
rotate_right_no_parent_fix(p, p_left);
NodeTraits::set_parent(p_left, p_parent);
set_child(header, p_left, p_parent, p_was_left);
}

private:

static void subtree_to_vine(node_ptr vine_tail, std::size_t &size)
{
std::size_t len = 0;
node_ptr remainder = NodeTraits::get_right(vine_tail);
while(remainder){
node_ptr tempptr = NodeTraits::get_left(remainder);
if(!tempptr){   
vine_tail = remainder;
remainder = NodeTraits::get_right(remainder);
++len;
}
else{ 
NodeTraits::set_left(remainder, NodeTraits::get_right(tempptr));
NodeTraits::set_right(tempptr, remainder);
remainder = tempptr;
NodeTraits::set_right(vine_tail, tempptr);
}
}
size = len;
}

static void compress_subtree(node_ptr scanner, std::size_t count)
{
while(count--){   
node_ptr child = NodeTraits::get_right(scanner);
node_ptr child_right = NodeTraits::get_right(child);
NodeTraits::set_right(scanner, child_right);
scanner = child_right;
node_ptr scanner_left = NodeTraits::get_left(scanner);
NodeTraits::set_right(child, scanner_left);
if(scanner_left)
NodeTraits::set_parent(scanner_left, child);
NodeTraits::set_left(scanner, child);
NodeTraits::set_parent(child, scanner);
}
}

static void vine_to_subtree(node_ptr super_root, std::size_t count)
{
const std::size_t one_szt = 1u;
std::size_t leaf_nodes = count + one_szt - std::size_t(one_szt << detail::floor_log2(count + one_szt));
compress_subtree(super_root, leaf_nodes);  
std::size_t vine_nodes = count - leaf_nodes;
while(vine_nodes > 1){
vine_nodes /= 2;
compress_subtree(super_root, vine_nodes);
}

for ( node_ptr q = super_root, p = NodeTraits::get_right(super_root)
; p
; q = p, p = NodeTraits::get_right(p)){
NodeTraits::set_parent(p, q);
}
}

static node_ptr get_root(const node_ptr & node)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT((!inited(node)));
node_ptr x = NodeTraits::get_parent(node);
if(x){
while(!base_type::is_header(x)){
x = NodeTraits::get_parent(x);
}
return x;
}
else{
return node;
}
}

template <class Cloner, class Disposer>
static node_ptr clone_subtree
(const const_node_ptr &source_parent, node_ptr target_parent
, Cloner cloner, Disposer disposer
, node_ptr &leftmost_out, node_ptr &rightmost_out
)
{
node_ptr target_sub_root = target_parent;
node_ptr source_root = NodeTraits::get_parent(source_parent);
if(!source_root){
leftmost_out = rightmost_out = source_root;
}
else{
node_ptr current = source_root;
node_ptr insertion_point = target_sub_root = cloner(current);

node_ptr leftmost  = target_sub_root;
node_ptr rightmost = target_sub_root;

NodeTraits::set_left(target_sub_root, node_ptr());
NodeTraits::set_right(target_sub_root, node_ptr());
NodeTraits::set_parent(target_sub_root, target_parent);

dispose_subtree_disposer<Disposer> rollback(disposer, target_sub_root);
while(true) {
if( NodeTraits::get_left(current) &&
!NodeTraits::get_left(insertion_point)) {
current = NodeTraits::get_left(current);
node_ptr temp = insertion_point;
insertion_point = cloner(current);
NodeTraits::set_left  (insertion_point, node_ptr());
NodeTraits::set_right (insertion_point, node_ptr());
NodeTraits::set_parent(insertion_point, temp);
NodeTraits::set_left  (temp, insertion_point);
if(rightmost == target_sub_root)
leftmost = insertion_point;
}
else if( NodeTraits::get_right(current) &&
!NodeTraits::get_right(insertion_point)){
current = NodeTraits::get_right(current);
node_ptr temp = insertion_point;
insertion_point = cloner(current);
NodeTraits::set_left  (insertion_point, node_ptr());
NodeTraits::set_right (insertion_point, node_ptr());
NodeTraits::set_parent(insertion_point, temp);
NodeTraits::set_right (temp, insertion_point);
rightmost = insertion_point;
}
else if(current == source_root){
break;
}
else{
current = NodeTraits::get_parent(current);
insertion_point = NodeTraits::get_parent(insertion_point);
}
}
rollback.release();
leftmost_out   = leftmost;
rightmost_out  = rightmost;
}
return target_sub_root;
}

template<class Disposer>
static void dispose_subtree(node_ptr x, Disposer disposer)
{
while (x){
node_ptr save(NodeTraits::get_left(x));
if (save) {
NodeTraits::set_left(x, NodeTraits::get_right(save));
NodeTraits::set_right(save, x);
}
else {
save = NodeTraits::get_right(x);
init(x);
disposer(x);
}
x = save;
}
}

template<class KeyType, class KeyNodePtrCompare>
static node_ptr lower_bound_loop
(node_ptr x, node_ptr y, const KeyType &key, KeyNodePtrCompare comp)
{
while(x){
if(comp(x, key)){
x = NodeTraits::get_right(x);
}
else{
y = x;
x = NodeTraits::get_left(x);
}
}
return y;
}

template<class KeyType, class KeyNodePtrCompare>
static node_ptr upper_bound_loop
(node_ptr x, node_ptr y, const KeyType &key, KeyNodePtrCompare comp)
{
while(x){
if(comp(key, x)){
y = x;
x = NodeTraits::get_left(x);
}
else{
x = NodeTraits::get_right(x);
}
}
return y;
}

template<class Checker>
static void check_subtree(const const_node_ptr& node, Checker checker, typename Checker::return_type& check_return)
{
const_node_ptr left = NodeTraits::get_left(node);
const_node_ptr right = NodeTraits::get_right(node);
typename Checker::return_type check_return_left;
typename Checker::return_type check_return_right;
if (left)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT(NodeTraits::get_parent(left) == node);
check_subtree(left, checker, check_return_left);
}
if (right)
{
BOOST_INTRUSIVE_INVARIANT_ASSERT(NodeTraits::get_parent(right) == node);
check_subtree(right, checker, check_return_right);
}
checker(node, check_return_left, check_return_right, check_return);
}
};


template<class NodeTraits>
struct get_algo<BsTreeAlgorithms, NodeTraits>
{
typedef bstree_algorithms<NodeTraits> type;
};

template <class ValueTraits, class NodePtrCompare, class ExtraChecker>
struct get_node_checker<BsTreeAlgorithms, ValueTraits, NodePtrCompare, ExtraChecker>
{
typedef detail::bstree_node_checker<ValueTraits, NodePtrCompare, ExtraChecker> type;
};


}  
}  

#include <boost/intrusive/detail/config_end.hpp>

#endif 
