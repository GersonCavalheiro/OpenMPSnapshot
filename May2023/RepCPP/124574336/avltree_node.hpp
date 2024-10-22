
#ifndef BOOST_INTRUSIVE_AVLTREE_NODE_HPP
#define BOOST_INTRUSIVE_AVLTREE_NODE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/pointer_rebind.hpp>
#include <boost/intrusive/avltree_algorithms.hpp>
#include <boost/intrusive/pointer_plus_bits.hpp>
#include <boost/intrusive/detail/mpl.hpp>

namespace boost {
namespace intrusive {


template<class VoidPointer>
struct compact_avltree_node
{
typedef typename pointer_rebind<VoidPointer, compact_avltree_node<VoidPointer> >::type       node_ptr;
typedef typename pointer_rebind<VoidPointer, const compact_avltree_node<VoidPointer> >::type const_node_ptr;
enum balance { negative_t, zero_t, positive_t };
node_ptr parent_, left_, right_;
};

template<class VoidPointer>
struct avltree_node
{
typedef typename pointer_rebind<VoidPointer, avltree_node<VoidPointer> >::type         node_ptr;
typedef typename pointer_rebind<VoidPointer, const avltree_node<VoidPointer> >::type   const_node_ptr;
enum balance { negative_t, zero_t, positive_t };
node_ptr parent_, left_, right_;
balance balance_;
};

template<class VoidPointer>
struct default_avltree_node_traits_impl
{
typedef avltree_node<VoidPointer>      node;
typedef typename node::node_ptr        node_ptr;
typedef typename node::const_node_ptr  const_node_ptr;

typedef typename node::balance balance;

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

BOOST_INTRUSIVE_FORCEINLINE static balance get_balance(const const_node_ptr & n)
{  return n->balance_;  }

BOOST_INTRUSIVE_FORCEINLINE static balance get_balance(const node_ptr & n)
{  return n->balance_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_balance(const node_ptr & n, balance b)
{  n->balance_ = b;  }

BOOST_INTRUSIVE_FORCEINLINE static balance negative()
{  return node::negative_t;  }

BOOST_INTRUSIVE_FORCEINLINE static balance zero()
{  return node::zero_t;  }

BOOST_INTRUSIVE_FORCEINLINE static balance positive()
{  return node::positive_t;  }
};

template<class VoidPointer>
struct compact_avltree_node_traits_impl
{
typedef compact_avltree_node<VoidPointer> node;
typedef typename node::node_ptr           node_ptr;
typedef typename node::const_node_ptr     const_node_ptr;
typedef typename node::balance balance;

typedef pointer_plus_bits<node_ptr, 2> ptr_bit;

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_parent(const const_node_ptr & n)
{  return ptr_bit::get_pointer(n->parent_);  }

BOOST_INTRUSIVE_FORCEINLINE static void set_parent(node_ptr n, node_ptr p)
{  ptr_bit::set_pointer(n->parent_, p);  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_left(const const_node_ptr & n)
{  return n->left_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_left(node_ptr n, node_ptr l)
{  n->left_ = l;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_right(const const_node_ptr & n)
{  return n->right_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_right(node_ptr n, node_ptr r)
{  n->right_ = r;  }

BOOST_INTRUSIVE_FORCEINLINE static balance get_balance(const const_node_ptr & n)
{  return (balance)ptr_bit::get_bits(n->parent_);  }

BOOST_INTRUSIVE_FORCEINLINE static void set_balance(const node_ptr & n, balance b)
{  ptr_bit::set_bits(n->parent_, (std::size_t)b);  }

BOOST_INTRUSIVE_FORCEINLINE static balance negative()
{  return node::negative_t;  }

BOOST_INTRUSIVE_FORCEINLINE static balance zero()
{  return node::zero_t;  }

BOOST_INTRUSIVE_FORCEINLINE static balance positive()
{  return node::positive_t;  }
};

template<class VoidPointer, bool Compact>
struct avltree_node_traits_dispatch
:  public default_avltree_node_traits_impl<VoidPointer>
{};

template<class VoidPointer>
struct avltree_node_traits_dispatch<VoidPointer, true>
:  public compact_avltree_node_traits_impl<VoidPointer>
{};

template<class VoidPointer, bool OptimizeSize = false>
struct avltree_node_traits
:  public avltree_node_traits_dispatch
< VoidPointer
, OptimizeSize &&
max_pointer_plus_bits
< VoidPointer
, detail::alignment_of<compact_avltree_node<VoidPointer> >::value
>::value >= 2u
>
{};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
