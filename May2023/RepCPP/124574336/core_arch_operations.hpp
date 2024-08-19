


#ifndef BOOST_ATOMIC_DETAIL_CORE_ARCH_OPERATIONS_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_ARCH_OPERATIONS_HPP_INCLUDED_

#include <boost/atomic/detail/core_arch_operations_fwd.hpp>
#include <boost/atomic/detail/core_operations_emulated.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/platform.hpp>
#include <boost/atomic/detail/storage_traits.hpp>

#if defined(BOOST_ATOMIC_DETAIL_CORE_ARCH_BACKEND_HEADER)
#include BOOST_ATOMIC_DETAIL_CORE_ARCH_BACKEND_HEADER(boost/atomic/detail/core_arch_ops_)
#endif

#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

template< std::size_t Size, bool Signed, bool Interprocess >
struct core_arch_operations :
public core_operations_emulated< Size, storage_traits< Size >::alignment, Signed, Interprocess >
{
};

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
