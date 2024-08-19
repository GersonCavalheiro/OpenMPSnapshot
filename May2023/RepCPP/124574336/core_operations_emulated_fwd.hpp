


#ifndef BOOST_ATOMIC_DETAIL_CORE_OPERATIONS_EMULATED_FWD_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CORE_OPERATIONS_EMULATED_FWD_HPP_INCLUDED_

#include <cstddef>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

template< std::size_t Size, std::size_t Alignment, bool Signed, bool Interprocess >
struct core_operations_emulated;

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
