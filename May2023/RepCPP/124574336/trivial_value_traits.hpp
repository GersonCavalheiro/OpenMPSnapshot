
#ifndef BOOST_INTRUSIVE_TRIVIAL_VALUE_TRAITS_HPP
#define BOOST_INTRUSIVE_TRIVIAL_VALUE_TRAITS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/pointer_traits.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template<class NodeTraits, link_mode_type LinkMode
#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
= safe_link
#endif
>
struct trivial_value_traits
{
typedef NodeTraits                                          node_traits;
typedef typename node_traits::node_ptr                      node_ptr;
typedef typename node_traits::const_node_ptr                const_node_ptr;
typedef typename node_traits::node                          value_type;
typedef node_ptr                                            pointer;
typedef const_node_ptr                                      const_pointer;
static const link_mode_type link_mode = LinkMode;
BOOST_INTRUSIVE_FORCEINLINE static node_ptr       to_node_ptr (value_type &value)
{  return pointer_traits<node_ptr>::pointer_to(value);  }
BOOST_INTRUSIVE_FORCEINLINE static const_node_ptr to_node_ptr (const value_type &value)
{  return pointer_traits<const_node_ptr>::pointer_to(value);  }
BOOST_INTRUSIVE_FORCEINLINE static const pointer  &      to_value_ptr(const node_ptr &n)        {  return n; }
BOOST_INTRUSIVE_FORCEINLINE static const const_pointer  &to_value_ptr(const const_node_ptr &n)  {  return n; }
};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
