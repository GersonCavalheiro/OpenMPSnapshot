
#ifndef BOOST_CONTAINER_ALLOCATOR_SCOPED_ALLOCATOR_FWD_HPP
#define BOOST_CONTAINER_ALLOCATOR_SCOPED_ALLOCATOR_FWD_HPP


#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/detail/std_fwd.hpp>
#include <boost/container/uses_allocator_fwd.hpp>

#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)
#include <boost/move/detail/fwd_macros.hpp>
#endif

namespace boost { namespace container {

#ifndef BOOST_CONTAINER_DOXYGEN_INVOKED

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

#if !defined(BOOST_CONTAINER_UNIMPLEMENTED_PACK_EXPANSION_TO_FIXED_LIST)

template <typename OuterAlloc, typename ...InnerAllocs>
class scoped_allocator_adaptor;

#else 

template <typename ...InnerAllocs>
class scoped_allocator_adaptor;

template <typename OuterAlloc, typename ...InnerAllocs>
class scoped_allocator_adaptor<OuterAlloc, InnerAllocs...>;

#endif   

#else    

template <typename OuterAlloc, BOOST_MOVE_CLASSDFLT9>
class scoped_allocator_adaptor;

#endif


#else    

#endif   

}} 

#include <boost/container/detail/config_end.hpp>

#endif 
