


#ifndef BOOST_ATOMIC_DETAIL_FENCE_OPERATIONS_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_FENCE_OPERATIONS_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/platform.hpp>

#if defined(BOOST_ATOMIC_DETAIL_CORE_BACKEND_HEADER)
#include BOOST_ATOMIC_DETAIL_CORE_BACKEND_HEADER(boost/atomic/detail/fence_ops_)
#else
#include <boost/atomic/detail/fence_arch_operations.hpp>

namespace boost {
namespace atomics {
namespace detail {

typedef fence_arch_operations fence_operations;

} 
} 
} 

#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#endif 
