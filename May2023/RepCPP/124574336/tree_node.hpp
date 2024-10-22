
#ifndef BOOST_INTRUSIVE_TREE_NODE_HPP
#define BOOST_INTRUSIVE_TREE_NODE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/pointer_rebind.hpp>

namespace boost {
namespace intrusive {

template<class VoidPointer>
struct tree_node
{
typedef typename pointer_rebind<VoidPointer, tree_node>::type  node_ptr;

node_ptr parent_, left_, right_;
};

template<class VoidPointer>
struct tree_node_traits
{
typedef tree_node<VoidPointer> node;

typedef typename node::node_ptr   node_ptr;
typedef typename pointer_rebind<VoidPointer, const node>::type const_node_ptr;

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_parent(const const_node_ptr & n)
{  return n->parent_;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_parent(const node_ptr & n)
{  return n->parent_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_parent(node_ptr n, node_ptr p)
{  n->parent_ = p;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_left(const const_node_ptr & n)
{  return n->left_;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_left(const node_ptr & n)
{  return n->left_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_left(node_ptr n, node_ptr l)
{  n->left_ = l;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_right(const const_node_ptr & n)
{  return n->right_;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_right(const node_ptr & n)
{  return n->right_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_right(node_ptr n, node_ptr r)
{  n->right_ = r;  }
};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
