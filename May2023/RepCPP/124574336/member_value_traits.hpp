
#ifndef BOOST_INTRUSIVE_MEMBER_VALUE_TRAITS_HPP
#define BOOST_INTRUSIVE_MEMBER_VALUE_TRAITS_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/detail/parent_from_member.hpp>
#include <boost/move/detail/to_raw_pointer.hpp> 
#include <boost/intrusive/pointer_traits.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template< class T, class NodeTraits
, typename NodeTraits::node T::* PtrToMember
, link_mode_type LinkMode
#ifdef BOOST_INTRUSIVE_DOXYGEN_INVOKED
= safe_link
#endif
>
struct member_value_traits
{
public:
typedef NodeTraits                                                   node_traits;
typedef T                                                            value_type;
typedef typename node_traits::node                                   node;
typedef typename node_traits::node_ptr                               node_ptr;
typedef typename node_traits::const_node_ptr                         const_node_ptr;
typedef pointer_traits<node_ptr>                                     node_ptr_traits;
typedef typename pointer_traits<node_ptr>::template
rebind_pointer<T>::type                                           pointer;
typedef typename pointer_traits<node_ptr>::template
rebind_pointer<const T>::type                                     const_pointer;
typedef value_type &                                                 reference;
typedef const value_type &                                           const_reference;
static const link_mode_type link_mode = LinkMode;

BOOST_INTRUSIVE_FORCEINLINE static node_ptr to_node_ptr(reference value)
{  return pointer_traits<node_ptr>::pointer_to(value.*PtrToMember);   }

BOOST_INTRUSIVE_FORCEINLINE static const_node_ptr to_node_ptr(const_reference value)
{  return pointer_traits<const_node_ptr>::pointer_to(value.*PtrToMember);   }

BOOST_INTRUSIVE_FORCEINLINE static pointer to_value_ptr(const node_ptr &n)
{
return pointer_traits<pointer>::pointer_to(*detail::parent_from_member<value_type, node>
(boost::movelib::to_raw_pointer(n), PtrToMember));
}

BOOST_INTRUSIVE_FORCEINLINE static const_pointer to_value_ptr(const const_node_ptr &n)
{
return pointer_traits<const_pointer>::pointer_to(*detail::parent_from_member<value_type, node>
(boost::movelib::to_raw_pointer(n), PtrToMember));

}
};

} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
