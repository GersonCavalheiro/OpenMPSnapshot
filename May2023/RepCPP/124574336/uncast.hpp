
#ifndef BOOST_INTRUSIVE_DETAIL_UNCAST_HPP
#define BOOST_INTRUSIVE_DETAIL_UNCAST_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/pointer_traits.hpp>
#include <boost/intrusive/detail/mpl.hpp>

namespace boost {
namespace intrusive {
namespace detail {

template<class ConstNodePtr>
struct uncast_types
{
typedef typename pointer_traits<ConstNodePtr>::element_type element_type;
typedef typename remove_const<element_type>::type           non_const_type;
typedef typename pointer_traits<ConstNodePtr>::
template rebind_pointer<non_const_type>::type            non_const_pointer;
typedef pointer_traits<non_const_pointer>                   non_const_traits;
};

template<class ConstNodePtr>
static typename uncast_types<ConstNodePtr>::non_const_pointer
uncast(const ConstNodePtr & ptr)
{
return uncast_types<ConstNodePtr>::non_const_traits::const_cast_from(ptr);
}

} 
} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
