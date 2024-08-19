


#ifndef BOOST_ATOMIC_DETAIL_WAIT_OPERATIONS_FWD_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_WAIT_OPERATIONS_FWD_HPP_INCLUDED_

#include <cstddef>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

template<
typename Base,
std::size_t Size = sizeof(typename Base::storage_type),
bool AlwaysLockFree = Base::is_always_lock_free,
bool Interprocess = Base::is_interprocess
>
struct wait_operations;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
