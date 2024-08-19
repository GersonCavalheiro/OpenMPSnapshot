


#ifndef BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPERATIONS_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_ARCH_OPERATIONS_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/platform.hpp>

#if defined(BOOST_ATOMIC_DETAIL_CORE_ARCH_BACKEND_HEADER)
#include BOOST_ATOMIC_DETAIL_CORE_ARCH_BACKEND_HEADER(boost/atomic/detail/fence_arch_ops_)
#else
#include <boost/atomic/detail/fence_operations_emulated.hpp>

namespace boost {
namespace atomics {
namespace detail {

typedef fence_operations_emulated fence_arch_operations;

} 
} 
} 

#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#endif 
