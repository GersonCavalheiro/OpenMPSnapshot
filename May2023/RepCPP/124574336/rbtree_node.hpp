
#ifndef BOOST_INTRUSIVE_RBTREE_NODE_HPP
#define BOOST_INTRUSIVE_RBTREE_NODE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/pointer_rebind.hpp>
#include <boost/intrusive/rbtree_algorithms.hpp>
#include <boost/intrusive/pointer_plus_bits.hpp>
#include <boost/intrusive/detail/mpl.hpp>
#include <boost/intrusive/detail/tree_node.hpp>

namespace boost {
namespace intrusive {


template<class VoidPointer>
struct compact_rbtree_node
{
typedef compact_rbtree_node<VoidPointer> node;
typedef typename pointer_rebind<VoidPointer, node >::type         node_ptr;
typedef typename pointer_rebind<VoidPointer, const node >::type   const_node_ptr;
enum color { red_t, black_t };
node_ptr parent_, left_, right_;
};

template<class VoidPointer>
struct rbtree_node
{
typedef rbtree_node<VoidPointer> node;
typedef typename pointer_rebind<VoidPointer, node >::type         node_ptr;
typedef typename pointer_rebind<VoidPointer, const node >::type   const_node_ptr;

enum color { red_t, black_t };
node_ptr parent_, left_, right_;
color color_;
};

template<class VoidPointer>
struct default_rbtree_node_traits_impl
{
typedef rbtree_node<VoidPointer> node;
typedef typename node::node_ptr        node_ptr;
typedef typename node::const_node_ptr  const_node_ptr;

typedef typename node::color color;

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

BOOST_INTRUSIVE_FORCEINLINE static color get_color(const const_node_ptr & n)
{  return n->color_;  }

BOOST_INTRUSIVE_FORCEINLINE static color get_color(const node_ptr & n)
{  return n->color_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_color(const node_ptr & n, color c)
{  n->color_ = c;  }

BOOST_INTRUSIVE_FORCEINLINE static color black()
{  return node::black_t;  }

BOOST_INTRUSIVE_FORCEINLINE static color red()
{  return node::red_t;  }
};

template<class VoidPointer>
struct compact_rbtree_node_traits_impl
{
typedef compact_rbtree_node<VoidPointer> node;
typedef typename node::node_ptr        node_ptr;
typedef typename node::const_node_ptr  const_node_ptr;

typedef pointer_plus_bits<node_ptr, 1> ptr_bit;

typedef typename node::color color;

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_parent(const const_node_ptr & n)
{  return ptr_bit::get_pointer(n->parent_);  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_parent(const node_ptr & n)
{  return ptr_bit::get_pointer(n->parent_);  }

BOOST_INTRUSIVE_FORCEINLINE static void set_parent(node_ptr n, node_ptr p)
{  ptr_bit::set_pointer(n->parent_, p);  }

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

BOOST_INTRUSIVE_FORCEINLINE static color get_color(const const_node_ptr & n)
{  return (color)ptr_bit::get_bits(n->parent_);  }

BOOST_INTRUSIVE_FORCEINLINE static color get_color(const node_ptr & n)
{  return (color)ptr_bit::get_bits(n->parent_);  }

BOOST_INTRUSIVE_FORCEINLINE static void set_color(const node_ptr & n, color c)
{  ptr_bit::set_bits(n->parent_, c != 0);  }

BOOST_INTRUSIVE_FORCEINLINE static color black()
{  return node::black_t;  }

BOOST_INTRUSIVE_FORCEINLINE static color red()
{  return node::red_t;  }
};

template<class VoidPointer, bool Compact>
struct rbtree_node_traits_dispatch
:  public default_rbtree_node_traits_impl<VoidPointer>
{};

template<class VoidPointer>
struct rbtree_node_traits_dispatch<VoidPointer, true>
:  public compact_rbtree_node_traits_impl<VoidPointer>
{};

template<class VoidPointer, bool OptimizeSize = false>
struct rbtree_node_traits
:  public rbtree_node_traits_dispatch
< VoidPointer
,  OptimizeSize &&
(max_pointer_plus_bits
< VoidPointer
, detail::alignment_of<compact_rbtree_node<VoidPointer> >::value
>::value >= 1)
>
{};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
