


#ifndef BOOST_ATOMIC_ATOMIC_FLAG_HPP_INCLUDED_
#define BOOST_ATOMIC_ATOMIC_FLAG_HPP_INCLUDED_

#include <boost/atomic/capabilities.hpp>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/atomic_flag_impl.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {

typedef atomics::detail::atomic_flag_impl< false > atomic_flag;

} 

using atomics::atomic_flag;

} 

#include <boost/atomic/detail/footer.hpp>

#endif 
