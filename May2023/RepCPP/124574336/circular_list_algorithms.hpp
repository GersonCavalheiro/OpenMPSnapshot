
#ifndef BOOST_INTRUSIVE_CIRCULAR_LIST_ALGORITHMS_HPP
#define BOOST_INTRUSIVE_CIRCULAR_LIST_ALGORITHMS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/algo_type.hpp>
#include <boost/core/no_exceptions_support.hpp>
#include <cstddef>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template<class NodeTraits>
class circular_list_algorithms
{
public:
typedef typename NodeTraits::node            node;
typedef typename NodeTraits::node_ptr        node_ptr;
typedef typename NodeTraits::const_node_ptr  const_node_ptr;
typedef NodeTraits                           node_traits;

BOOST_INTRUSIVE_FORCEINLINE static void init(node_ptr this_node)
{
const node_ptr null_node = node_ptr();
NodeTraits::set_next(this_node, null_node);
NodeTraits::set_previous(this_node, null_node);
}

BOOST_INTRUSIVE_FORCEINLINE static bool inited(const const_node_ptr &this_node)
{  return !NodeTraits::get_next(this_node); }

BOOST_INTRUSIVE_FORCEINLINE static void init_header(node_ptr this_node)
{
NodeTraits::set_next(this_node, this_node);
NodeTraits::set_previous(this_node, this_node);
}


BOOST_INTRUSIVE_FORCEINLINE static bool unique(const const_node_ptr &this_node)
{
node_ptr next = NodeTraits::get_next(this_node);
return !next || next == this_node;
}

static std::size_t count(const const_node_ptr &this_node)
{
std::size_t result = 0;
const_node_ptr p = this_node;
do{
p = NodeTraits::get_next(p);
++result;
}while (p != this_node);
return result;
}

BOOST_INTRUSIVE_FORCEINLINE static node_ptr unlink(node_ptr this_node)
{
node_ptr next(NodeTraits::get_next(this_node));
node_ptr prev(NodeTraits::get_previous(this_node));
NodeTraits::set_next(prev, next);
NodeTraits::set_previous(next, prev);
return next;
}

BOOST_INTRUSIVE_FORCEINLINE static void unlink(node_ptr b, node_ptr e)
{
if (b != e) {
node_ptr prevb(NodeTraits::get_previous(b));
NodeTraits::set_previous(e, prevb);
NodeTraits::set_next(prevb, e);
}
}

BOOST_INTRUSIVE_FORCEINLINE static void link_before(node_ptr nxt_node, node_ptr this_node)
{
node_ptr prev(NodeTraits::get_previous(nxt_node));
NodeTraits::set_previous(this_node, prev);
NodeTraits::set_next(this_node, nxt_node);
NodeTraits::set_previous(nxt_node, this_node);
NodeTraits::set_next(prev, this_node);
}

BOOST_INTRUSIVE_FORCEINLINE static void link_after(node_ptr prev_node, node_ptr this_node)
{
node_ptr next(NodeTraits::get_next(prev_node));
NodeTraits::set_previous(this_node, prev_node);
NodeTraits::set_next(this_node, next);
NodeTraits::set_next(prev_node, this_node);
NodeTraits::set_previous(next, this_node);
}

static void swap_nodes(node_ptr this_node, node_ptr other_node)
{
if (other_node == this_node)
return;
bool this_inited  = inited(this_node);
bool other_inited = inited(other_node);
if(this_inited){
init_header(this_node);
}
if(other_inited){
init_header(other_node);
}

node_ptr next_this(NodeTraits::get_next(this_node));
node_ptr prev_this(NodeTraits::get_previous(this_node));
node_ptr next_other(NodeTraits::get_next(other_node));
node_ptr prev_other(NodeTraits::get_previous(other_node));
swap_prev(next_this, next_other);
swap_next(prev_this, prev_other);
swap_next(this_node, other_node);
swap_prev(this_node, other_node);

if(this_inited){
init(other_node);
}
if(other_inited){
init(this_node);
}
}

static void transfer(node_ptr p, node_ptr b, node_ptr e)
{
if (b != e) {
node_ptr prev_p(NodeTraits::get_previous(p));
node_ptr prev_b(NodeTraits::get_previous(b));
node_ptr prev_e(NodeTraits::get_previous(e));
NodeTraits::set_next(prev_e, p);
NodeTraits::set_previous(p, prev_e);
NodeTraits::set_next(prev_b, e);
NodeTraits::set_previous(e, prev_b);
NodeTraits::set_next(prev_p, b);
NodeTraits::set_previous(b, prev_p);
}
}

static void transfer(node_ptr p, node_ptr i)
{
node_ptr n(NodeTraits::get_next(i));
if(n != p && i != p){
node_ptr prev_p(NodeTraits::get_previous(p));
node_ptr prev_i(NodeTraits::get_previous(i));
NodeTraits::set_next(prev_p, i);
NodeTraits::set_previous(i, prev_p);
NodeTraits::set_next(i, p);
NodeTraits::set_previous(p, i);
NodeTraits::set_previous(n, prev_i);
NodeTraits::set_next(prev_i, n);

}
}

static void reverse(node_ptr p)
{
node_ptr f(NodeTraits::get_next(p));
node_ptr i(NodeTraits::get_next(f)), e(p);

while(i != e) {
node_ptr n = i;
i = NodeTraits::get_next(i);
transfer(f, n, i);
f = n;
}
}

static void move_backwards(node_ptr p, std::size_t n)
{
if(!n) return;
node_ptr first  = NodeTraits::get_next(p);
if(first == NodeTraits::get_previous(p)) return;
unlink(p);
while(n--){
first = NodeTraits::get_next(first);
}
link_before(first, p);
}

static void move_forward(node_ptr p, std::size_t n)
{
if(!n)   return;
node_ptr last  = NodeTraits::get_previous(p);
if(last == NodeTraits::get_next(p))   return;

unlink(p);
while(n--){
last = NodeTraits::get_previous(last);
}
link_after(last, p);
}

static std::size_t distance(const const_node_ptr &f, const const_node_ptr &l)
{
const_node_ptr i(f);
std::size_t result = 0;
while(i != l){
i = NodeTraits::get_next(i);
++result;
}
return result;
}

struct stable_partition_info
{
std::size_t num_1st_partition;
std::size_t num_2nd_partition;
node_ptr    beg_2st_partition;
};

template<class Pred>
static void stable_partition(node_ptr beg, node_ptr end, Pred pred, stable_partition_info &info)
{
node_ptr bcur = node_traits::get_previous(beg);
node_ptr cur  = beg;
node_ptr new_f = end;

std::size_t num1 = 0, num2 = 0;
while(cur != end){
if(pred(cur)){
++num1;
bcur = cur;
cur  = node_traits::get_next(cur);
}
else{
++num2;
node_ptr last_to_remove = bcur;
new_f = cur;
bcur = cur;
cur  = node_traits::get_next(cur);
BOOST_TRY{
while(cur != end){
if(pred(cur)){ 
++num1;
node_traits::set_next    (last_to_remove, cur);
node_traits::set_previous(cur, last_to_remove);
last_to_remove = cur;
node_ptr nxt = node_traits::get_next(cur);
node_traits::set_next    (bcur, nxt);
node_traits::set_previous(nxt, bcur);
cur = nxt;
}
else{
++num2;
bcur = cur;
cur  = node_traits::get_next(cur);
}
}
}
BOOST_CATCH(...){
node_traits::set_next    (last_to_remove, new_f);
node_traits::set_previous(new_f, last_to_remove);
BOOST_RETHROW;
}
BOOST_CATCH_END
node_traits::set_next(last_to_remove, new_f);
node_traits::set_previous(new_f, last_to_remove);
break;
}
}
info.num_1st_partition = num1;
info.num_2nd_partition = num2;
info.beg_2st_partition = new_f;
}

private:
BOOST_INTRUSIVE_FORCEINLINE static void swap_prev(node_ptr this_node, node_ptr other_node)
{
node_ptr temp(NodeTraits::get_previous(this_node));
NodeTraits::set_previous(this_node, NodeTraits::get_previous(other_node));
NodeTraits::set_previous(other_node, temp);
}

BOOST_INTRUSIVE_FORCEINLINE static void swap_next(node_ptr this_node, node_ptr other_node)
{
node_ptr temp(NodeTraits::get_next(this_node));
NodeTraits::set_next(this_node, NodeTraits::get_next(other_node));
NodeTraits::set_next(other_node, temp);
}
};


template<class NodeTraits>
struct get_algo<CircularListAlgorithms, NodeTraits>
{
typedef circular_list_algorithms<NodeTraits> type;
};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
