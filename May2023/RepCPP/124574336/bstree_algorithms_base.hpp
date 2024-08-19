
#ifndef BOOST_INTRUSIVE_BSTREE_ALGORITHMS_BASE_HPP
#define BOOST_INTRUSIVE_BSTREE_ALGORITHMS_BASE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/uncast.hpp>

namespace boost {
namespace intrusive {

template<class NodeTraits>
class bstree_algorithms_base
{
public:
typedef typename NodeTraits::node            node;
typedef NodeTraits                           node_traits;
typedef typename NodeTraits::node_ptr        node_ptr;
typedef typename NodeTraits::const_node_ptr  const_node_ptr;

static node_ptr next_node(const node_ptr & node)
{
node_ptr const n_right(NodeTraits::get_right(node));
if(n_right){
return minimum(n_right);
}
else {
node_ptr n(node);
node_ptr p(NodeTraits::get_parent(n));
while(n == NodeTraits::get_right(p)){
n = p;
p = NodeTraits::get_parent(p);
}
return NodeTraits::get_right(n) != p ? p : n;
}
}

static node_ptr prev_node(const node_ptr & node)
{
if(is_header(node)){
return maximum(NodeTraits::get_parent(node));
}
else if(NodeTraits::get_left(node)){
return maximum(NodeTraits::get_left(node));
}
else {
node_ptr p(node);
node_ptr x = NodeTraits::get_parent(p);
while(p == NodeTraits::get_left(x)){
p = x;
x = NodeTraits::get_parent(x);
}
return x;
}
}

static node_ptr minimum(node_ptr node)
{
for(node_ptr p_left = NodeTraits::get_left(node)
;p_left
;p_left = NodeTraits::get_left(node)){
node = p_left;
}
return node;
}

static node_ptr maximum(node_ptr node)
{
for(node_ptr p_right = NodeTraits::get_right(node)
;p_right
;p_right = NodeTraits::get_right(node)){
node = p_right;
}
return node;
}

static bool is_header(const const_node_ptr & p)
{
node_ptr p_left (NodeTraits::get_left(p));
node_ptr p_right(NodeTraits::get_right(p));
if(!NodeTraits::get_parent(p) || 
(p_left && p_right &&         
(p_left == p_right ||      
(NodeTraits::get_parent(p_left)  != p ||
NodeTraits::get_parent(p_right) != p ))
)){
return true;
}
return false;
}

static node_ptr get_header(const const_node_ptr & node)
{
node_ptr n(detail::uncast(node));
node_ptr p(NodeTraits::get_parent(node));
if(p){
node_ptr pp(NodeTraits::get_parent(p));
if(n != pp){
do{
n = p;
p = pp;
pp = NodeTraits::get_parent(pp);
}while(n != pp);
n = p;
}
else if(!bstree_algorithms_base::is_header(n)){
n = p;
}
}
return n;
}
};

}  
}  

#endif 
