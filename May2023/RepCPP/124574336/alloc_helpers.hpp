#ifndef BOOST_CONTAINER_DETAIL_ALLOC_TRAITS_HPP
#define BOOST_CONTAINER_DETAIL_ALLOC_TRAITS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/move/adl_move_swap.hpp>
#include <boost/move/utility_core.hpp>

namespace boost {
namespace container {
namespace dtl {

template<class AllocatorType>
BOOST_CONTAINER_FORCEINLINE void swap_alloc(AllocatorType &, AllocatorType &, dtl::false_type)
BOOST_NOEXCEPT_OR_NOTHROW
{}

template<class AllocatorType>
BOOST_CONTAINER_FORCEINLINE void swap_alloc(AllocatorType &l, AllocatorType &r, dtl::true_type)
{  boost::adl_move_swap(l, r);   }

template<class AllocatorType>
BOOST_CONTAINER_FORCEINLINE void assign_alloc(AllocatorType &, const AllocatorType &, dtl::false_type)
BOOST_NOEXCEPT_OR_NOTHROW
{}

template<class AllocatorType>
BOOST_CONTAINER_FORCEINLINE void assign_alloc(AllocatorType &l, const AllocatorType &r, dtl::true_type)
{  l = r;   }

template<class AllocatorType>
BOOST_CONTAINER_FORCEINLINE void move_alloc(AllocatorType &, AllocatorType &, dtl::false_type)
BOOST_NOEXCEPT_OR_NOTHROW
{}

template<class AllocatorType>
BOOST_CONTAINER_FORCEINLINE void move_alloc(AllocatorType &l, AllocatorType &r, dtl::true_type)
{  l = ::boost::move(r);   }

}  
}  
}  

#endif   
