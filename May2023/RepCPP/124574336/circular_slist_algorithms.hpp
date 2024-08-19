
#ifndef BOOST_INTRUSIVE_CIRCULAR_SLIST_ALGORITHMS_HPP
#define BOOST_INTRUSIVE_CIRCULAR_SLIST_ALGORITHMS_HPP

#include <cstddef>
#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/common_slist_algorithms.hpp>
#include <boost/intrusive/detail/algo_type.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template<class NodeTraits>
class circular_slist_algorithms
: public detail::common_slist_algorithms<NodeTraits>
{
typedef detail::common_slist_algorithms<NodeTraits> base_t;
public:
typedef typename NodeTraits::node            node;
typedef typename NodeTraits::node_ptr        node_ptr;
typedef typename NodeTraits::const_node_ptr  const_node_ptr;
typedef NodeTraits                           node_traits;

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)

static void init(node_ptr this_node);

static bool unique(const_node_ptr this_node);

static bool inited(const_node_ptr this_node);

static void unlink_after(node_ptr prev_node);

static void unlink_after(node_ptr prev_node, node_ptr last_node);

static void link_after(node_ptr prev_node, node_ptr this_node);

static void transfer_after(node_ptr p, node_ptr b, node_ptr e);

#endif   

BOOST_INTRUSIVE_FORCEINLINE static void init_header(node_ptr this_node)
{  NodeTraits::set_next(this_node, this_node);  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_previous_node(const node_ptr &prev_init_node, const node_ptr &this_node)
{  return base_t::get_previous_node(prev_init_node, this_node);   }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_previous_node(const node_ptr & this_node)
{  return base_t::get_previous_node(this_node, this_node); }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_previous_previous_node(const node_ptr & this_node)
{  return get_previous_previous_node(this_node, this_node); }

static node_ptr get_previous_previous_node(node_ptr p, const node_ptr & this_node)
{
node_ptr p_next = NodeTraits::get_next(p);
node_ptr p_next_next = NodeTraits::get_next(p_next);
while (this_node != p_next_next){
p = p_next;
p_next = p_next_next;
p_next_next = NodeTraits::get_next(p_next);
}
return p;
}

static std::size_t count(const const_node_ptr & this_node)
{
std::size_t result = 0;
const_node_ptr p = this_node;
do{
p = NodeTraits::get_next(p);
++result;
} while (p != this_node);
return result;
}

BOOST_INTRUSIVE_FORCEINLINE static void unlink(node_ptr this_node)
{
if(NodeTraits::get_next(this_node))
base_t::unlink_after(get_previous_node(this_node));
}

BOOST_INTRUSIVE_FORCEINLINE static void link_before (node_ptr nxt_node, node_ptr this_node)
{  base_t::link_after(get_previous_node(nxt_node), this_node);   }

static void swap_nodes(node_ptr this_node, node_ptr other_node)
{
if (other_node == this_node)
return;
const node_ptr this_next = NodeTraits::get_next(this_node);
const node_ptr other_next = NodeTraits::get_next(other_node);
const bool this_null   = !this_next;
const bool other_null  = !other_next;
const bool this_empty  = this_next == this_node;
const bool other_empty = other_next == other_node;

if(!(other_null || other_empty)){
NodeTraits::set_next(this_next == other_node ? other_node : get_previous_node(other_node), this_node );
}
if(!(this_null | this_empty)){
NodeTraits::set_next(other_next == this_node ? this_node  : get_previous_node(this_node), other_node );
}
NodeTraits::set_next(this_node,  other_empty ? this_node  : (other_next == this_node ? other_node : other_next) );
NodeTraits::set_next(other_node, this_empty  ? other_node : (this_next == other_node ? this_node :  this_next ) );
}

static void reverse(node_ptr p)
{
node_ptr i = NodeTraits::get_next(p), e(p);
for (;;) {
node_ptr nxt(NodeTraits::get_next(i));
if (nxt == e)
break;
base_t::transfer_after(e, i, nxt);
}
}

static node_ptr move_backwards(node_ptr p, std::size_t n)
{
if(!n) return node_ptr();
node_ptr first  = NodeTraits::get_next(p);

if(NodeTraits::get_next(first) == p)
return node_ptr();

bool end_found = false;
node_ptr new_last = node_ptr();

for(std::size_t i = 1; i <= n; ++i){
new_last = first;
first = NodeTraits::get_next(first);
if(first == p){
n %= i;
if(!n)
return node_ptr();
i = 0;
first = NodeTraits::get_next(p);
base_t::unlink_after(new_last);
end_found = true;
}
}

if(!end_found){
base_t::unlink_after(base_t::get_previous_node(first, p));
}

base_t::link_after(new_last, p);
return new_last;
}

static node_ptr move_forward(node_ptr p, std::size_t n)
{
if(!n) return node_ptr();
node_ptr first  = node_traits::get_next(p);

if(node_traits::get_next(first) == p) return node_ptr();

node_ptr old_last(first), next_to_it, new_last(p);
std::size_t distance = 1;
while(p != (next_to_it = node_traits::get_next(old_last))){
if(++distance > n)
new_last = node_traits::get_next(new_last);
old_last = next_to_it;
}
if(distance <= n){
std::size_t new_before_last_pos = (distance - (n % distance))% distance;
if(!new_before_last_pos)   return node_ptr();

for( new_last = p
; new_before_last_pos--
; new_last = node_traits::get_next(new_last)){
}
}

base_t::unlink_after(old_last);
base_t::link_after(new_last, p);
return new_last;
}
};


template<class NodeTraits>
struct get_algo<CircularSListAlgorithms, NodeTraits>
{
typedef circular_slist_algorithms<NodeTraits> type;
};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
