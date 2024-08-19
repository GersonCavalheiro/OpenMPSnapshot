


#ifndef BOOST_ATOMIC_DETAIL_INTPTR_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_INTPTR_HPP_INCLUDED_

#include <boost/cstdint.hpp>
#if defined(BOOST_HAS_INTPTR_T)
#include <cstddef>
#endif
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

#if !defined(BOOST_HAS_INTPTR_T)
using boost::uintptr_t;
using boost::intptr_t;
#else
typedef std::size_t uintptr_t;
typedef std::ptrdiff_t intptr_t;
#endif

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
