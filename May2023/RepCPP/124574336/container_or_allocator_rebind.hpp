#ifndef BOOST_CONTAINER_DETAIL_CONTAINER_OR_ALLOCATOR_REBIND_HPP
#define BOOST_CONTAINER_DETAIL_CONTAINER_OR_ALLOCATOR_REBIND_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/allocator_traits.hpp>
#include <boost/container/detail/container_rebind.hpp>
#include <boost/container/detail/is_container.hpp>

namespace boost {
namespace container {
namespace dtl {

template<class AllocatorOrContainer, class ToType, bool = is_container<AllocatorOrContainer>::value>
struct container_or_allocator_rebind_impl
: container_rebind<AllocatorOrContainer, ToType>
{};

template<class AllocatorOrContainer, class ToType>
struct container_or_allocator_rebind_impl<AllocatorOrContainer, ToType, false>
: allocator_traits<AllocatorOrContainer>::template portable_rebind_alloc<ToType>
{};

template<class ToType>
struct container_or_allocator_rebind_impl<void, ToType, false>
: real_allocator<ToType, void>
{};

template<class AllocatorOrContainer, class ToType>
struct container_or_allocator_rebind
: container_or_allocator_rebind_impl<AllocatorOrContainer, ToType>
{};

}  
}  
}  

#endif   
