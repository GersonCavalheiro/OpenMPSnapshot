
#ifndef BOOST_INTRUSIVE_DETAIL_DEFAULT_HEADER_HOLDER_HPP
#define BOOST_INTRUSIVE_DETAIL_DEFAULT_HEADER_HOLDER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/move/detail/to_raw_pointer.hpp>

namespace boost {
namespace intrusive {
namespace detail {

template < typename NodeTraits >
struct default_header_holder : public NodeTraits::node
{
typedef NodeTraits node_traits;
typedef typename node_traits::node node;
typedef typename node_traits::node_ptr node_ptr;
typedef typename node_traits::const_node_ptr const_node_ptr;

default_header_holder() : node() {}

BOOST_INTRUSIVE_FORCEINLINE const_node_ptr get_node() const
{ return pointer_traits< const_node_ptr >::pointer_to(*static_cast< const node* >(this)); }

BOOST_INTRUSIVE_FORCEINLINE node_ptr get_node()
{ return pointer_traits< node_ptr >::pointer_to(*static_cast< node* >(this)); }

BOOST_INTRUSIVE_FORCEINLINE static default_header_holder* get_holder(const node_ptr &p)
{ return static_cast< default_header_holder* >(boost::movelib::to_raw_pointer(p)); }
};

template < typename ValueTraits, typename HeaderHolder >
struct get_header_holder_type
{
typedef HeaderHolder type;
};
template < typename ValueTraits >
struct get_header_holder_type< ValueTraits, void >
{
typedef default_header_holder< typename ValueTraits::node_traits > type;
};

} 
} 
} 

#endif 
