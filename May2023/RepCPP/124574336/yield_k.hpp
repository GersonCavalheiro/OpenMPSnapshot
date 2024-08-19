#ifndef BOOST_SMART_PTR_DETAIL_YIELD_K_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_YIELD_K_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/smart_ptr/detail/sp_thread_pause.hpp>
#include <boost/smart_ptr/detail/sp_thread_sleep.hpp>
#include <boost/config.hpp>

namespace boost
{

namespace detail
{

inline void yield( unsigned k )
{

if( k == 0 )
{
sp_thread_pause();
}
else
{
sp_thread_sleep();
}
}

} 

} 

#endif 
