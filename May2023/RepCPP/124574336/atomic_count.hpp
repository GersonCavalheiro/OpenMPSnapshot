#ifndef BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_ATOMIC_COUNT_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_has_gcc_intrinsics.hpp>
#include <boost/smart_ptr/detail/sp_has_sync_intrinsics.hpp>
#include <boost/config.hpp>

#if defined( BOOST_AC_DISABLE_THREADS )
# include <boost/smart_ptr/detail/atomic_count_nt.hpp>

#elif defined( BOOST_AC_USE_STD_ATOMIC )
# include <boost/smart_ptr/detail/atomic_count_std_atomic.hpp>

#elif defined( BOOST_AC_USE_SPINLOCK )
# include <boost/smart_ptr/detail/atomic_count_spin.hpp>

#elif defined( BOOST_AC_USE_PTHREADS )
# include <boost/smart_ptr/detail/atomic_count_pt.hpp>

#elif defined( BOOST_SP_DISABLE_THREADS )
# include <boost/smart_ptr/detail/atomic_count_nt.hpp>

#elif defined( BOOST_SP_USE_STD_ATOMIC )
# include <boost/smart_ptr/detail/atomic_count_std_atomic.hpp>

#elif defined( BOOST_SP_USE_SPINLOCK )
# include <boost/smart_ptr/detail/atomic_count_spin.hpp>

#elif defined( BOOST_SP_USE_PTHREADS )
# include <boost/smart_ptr/detail/atomic_count_pt.hpp>

#elif defined( BOOST_DISABLE_THREADS ) && !defined( BOOST_SP_ENABLE_THREADS ) && !defined( BOOST_DISABLE_WIN32 )
# include <boost/smart_ptr/detail/atomic_count_nt.hpp>

#elif defined( BOOST_SP_HAS_GCC_INTRINSICS )
# include <boost/smart_ptr/detail/atomic_count_gcc_atomic.hpp>

#elif !defined( BOOST_NO_CXX11_HDR_ATOMIC )
# include <boost/smart_ptr/detail/atomic_count_std_atomic.hpp>

#elif defined( BOOST_SP_HAS_SYNC_INTRINSICS )
# include <boost/smart_ptr/detail/atomic_count_sync.hpp>

#elif defined( __GNUC__ ) && ( defined( __i386__ ) || defined( __x86_64__ ) ) && !defined( __PATHSCALE__ )
# include <boost/smart_ptr/detail/atomic_count_gcc_x86.hpp>

#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
# include <boost/smart_ptr/detail/atomic_count_win32.hpp>

#elif defined(__GLIBCPP__) || defined(__GLIBCXX__)
# include <boost/smart_ptr/detail/atomic_count_gcc.hpp>

#elif !defined( BOOST_HAS_THREADS )
# include <boost/smart_ptr/detail/atomic_count_nt.hpp>

#else
# include <boost/smart_ptr/detail/atomic_count_spin.hpp>

#endif

#endif 
