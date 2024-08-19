#ifndef BOOST_SMART_PTR_DETAIL_SPINLOCK_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SPINLOCK_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_has_gcc_intrinsics.hpp>
#include <boost/smart_ptr/detail/sp_has_sync_intrinsics.hpp>
#include <boost/config.hpp>

#if defined( BOOST_SP_USE_STD_ATOMIC )
#  include <boost/smart_ptr/detail/spinlock_std_atomic.hpp>

#elif defined( BOOST_SP_USE_PTHREADS )
#  include <boost/smart_ptr/detail/spinlock_pt.hpp>

#elif defined( BOOST_SP_HAS_GCC_INTRINSICS )
#  include <boost/smart_ptr/detail/spinlock_gcc_atomic.hpp>

#elif !defined( BOOST_NO_CXX11_HDR_ATOMIC )
#  include <boost/smart_ptr/detail/spinlock_std_atomic.hpp>

#elif defined(__GNUC__) && defined( __arm__ ) && !defined( __thumb__ )
#  include <boost/smart_ptr/detail/spinlock_gcc_arm.hpp>

#elif defined( BOOST_SP_HAS_SYNC_INTRINSICS )
#  include <boost/smart_ptr/detail/spinlock_sync.hpp>

#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#  include <boost/smart_ptr/detail/spinlock_w32.hpp>

#elif defined(BOOST_HAS_PTHREADS)
#  include <boost/smart_ptr/detail/spinlock_pt.hpp>

#elif !defined(BOOST_HAS_THREADS)
#  include <boost/smart_ptr/detail/spinlock_nt.hpp>

#else
#  error Unrecognized threading platform
#endif

#endif 
