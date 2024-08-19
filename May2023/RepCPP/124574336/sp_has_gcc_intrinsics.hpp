#ifndef BOOST_SMART_PTR_DETAIL_SP_HAS_GCC_INTRINSICS_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_HAS_GCC_INTRINSICS_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif




#if defined( __ATOMIC_RELAXED ) && defined( __ATOMIC_ACQUIRE ) && defined( __ATOMIC_RELEASE ) && defined( __ATOMIC_ACQ_REL )

# define BOOST_SP_HAS_GCC_INTRINSICS

#endif

#endif 
