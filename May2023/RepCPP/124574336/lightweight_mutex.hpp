#ifndef BOOST_SMART_PTR_DETAIL_LIGHTWEIGHT_MUTEX_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_LIGHTWEIGHT_MUTEX_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>

#if !defined(BOOST_NO_CXX11_HDR_MUTEX )
#  include <boost/smart_ptr/detail/lwm_std_mutex.hpp>
#elif defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#  include <boost/smart_ptr/detail/lwm_win32_cs.hpp>
#else
#  include <boost/smart_ptr/detail/lwm_pthreads.hpp>
#endif

#endif 
