
#ifndef BOOST_INTRUSIVE_LINEAR_SLIST_ALGORITHMS_HPP
#define BOOST_INTRUSIVE_LINEAR_SLIST_ALGORITHMS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/detail/common_slist_algorithms.hpp>
#include <boost/intrusive/detail/algo_type.hpp>
#include <cstddef>
#include <boost/intrusive/detail/minimal_pair_header.hpp>   

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template<class NodeTraits>
class linear_slist_algorithms
: public detail::common_slist_algorithms<NodeTraits>
{
typedef detail::common_slist_algorithms<NodeTraits> base_t;
public:
typedef typename NodeTraits::node            node;
typedef typename NodeTraits::node_ptr        node_ptr;
typedef typename NodeTraits::const_node_ptr  const_node_ptr;
typedef NodeTraits                           node_traits;

#if defined(BOOST_INTRUSIVE_DOXYGEN_INVOKED)

static void init(const node_ptr & this_node);

static bool unique(const_node_ptr this_node);

static bool inited(const_node_ptr this_node);

static void unlink_after(const node_ptr & prev_node);

static void unlink_after(const node_ptr & prev_node, const node_ptr & last_node);

static void link_after(const node_ptr & prev_node, const node_ptr & this_node);

static void transfer_after(const node_ptr & p, const node_ptr & b, const node_ptr & e);

#endif   

BOOST_INTRUSIVE_FORCEINLINE static void init_header(const node_ptr & this_node)
{  NodeTraits::set_next(this_node, node_ptr ());  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_previous_node(const node_ptr & prev_init_node, const node_ptr & this_node)
{  return base_t::get_previous_node(prev_init_node, this_node);   }

static std::size_t count(const const_node_ptr & this_node)
{
std::size_t result = 0;
const_node_ptr p = this_node;
do{
p = NodeTraits::get_next(p);
++result;
} while (p);
return result;
}

static void swap_trailing_nodes(node_ptr this_node, node_ptr other_node)
{
node_ptr this_nxt    = NodeTraits::get_next(this_node);
node_ptr other_nxt   = NodeTraits::get_next(other_node);
NodeTraits::set_next(this_node, other_nxt);
NodeTraits::set_next(other_node, this_nxt);
}

static node_ptr reverse(node_ptr p)
{
if(!p) return node_ptr();
node_ptr i = NodeTraits::get_next(p);
node_ptr first(p);
while(i){
node_ptr nxti(NodeTraits::get_next(i));
base_t::unlink_after(p);
NodeTraits::set_next(i, first);
first = i;
i = nxti;
}
return first;
}

static std::pair<node_ptr, node_ptr> move_first_n_backwards(node_ptr p, std::size_t n)
{
std::pair<node_ptr, node_ptr> ret;
if(!n || !p || !NodeTraits::get_next(p)){
return ret;
}

node_ptr first = p;
bool end_found = false;
node_ptr new_last = node_ptr();
node_ptr old_last = node_ptr();

for(std::size_t i = 1; i <= n; ++i){
new_last = first;
first = NodeTraits::get_next(first);
if(first == node_ptr()){
n %= i;
if(!n)   return ret;
old_last = new_last;
i = 0;
first = p;
end_found = true;
}
}

if(!end_found){
old_last = base_t::get_previous_node(first, node_ptr());
}

NodeTraits::set_next(old_last, p);
NodeTraits::set_next(new_last, node_ptr());
ret.first   = first;
ret.second  = new_last;
return ret;
}

static std::pair<node_ptr, node_ptr> move_first_n_forward(node_ptr p, std::size_t n)
{
std::pair<node_ptr, node_ptr> ret;
if(!n || !p || !NodeTraits::get_next(p))
return ret;

node_ptr first  = p;

node_ptr old_last(first), next_to_it, new_last(p);
std::size_t distance = 1;
while(!!(next_to_it = node_traits::get_next(old_last))){
if(distance++ > n)
new_last = node_traits::get_next(new_last);
old_last = next_to_it;
}
if(distance <= n){
std::size_t new_before_last_pos = (distance - (n % distance))% distance;
if(!new_before_last_pos)
return ret;

for( new_last = p
; --new_before_last_pos
; new_last = node_traits::get_next(new_last)){
}
}

node_ptr new_first(node_traits::get_next(new_last));
NodeTraits::set_next(old_last, p);
NodeTraits::set_next(new_last, node_ptr());
ret.first   = new_first;
ret.second  = new_last;
return ret;
}
};


template<class NodeTraits>
struct get_algo<LinearSListAlgorithms, NodeTraits>
{
typedef linear_slist_algorithms<NodeTraits> type;
};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
