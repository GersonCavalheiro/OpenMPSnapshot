
#ifndef BOOST_INTRUSIVE_LIST_NODE_HPP
#define BOOST_INTRUSIVE_LIST_NODE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/pointer_rebind.hpp>

namespace boost {
namespace intrusive {


template<class VoidPointer>
struct list_node
{
typedef typename pointer_rebind<VoidPointer, list_node>::type  node_ptr;
node_ptr next_;
node_ptr prev_;
};

template<class VoidPointer>
struct list_node_traits
{
typedef list_node<VoidPointer>      node;
typedef typename node::node_ptr     node_ptr;
typedef typename pointer_rebind<VoidPointer, const node>::type   const_node_ptr;

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_previous(const const_node_ptr & n)
{  return n->prev_;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_previous(const node_ptr & n)
{  return n->prev_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_previous(node_ptr n, node_ptr prev)
{  n->prev_ = prev;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_next(const const_node_ptr & n)
{  return n->next_;  }

BOOST_INTRUSIVE_FORCEINLINE static node_ptr get_next(const node_ptr & n)
{  return n->next_;  }

BOOST_INTRUSIVE_FORCEINLINE static void set_next(node_ptr n, node_ptr next)
{  n->next_ = next;  }
};

} 
} 

#endif 
