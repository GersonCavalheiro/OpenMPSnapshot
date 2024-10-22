
#ifndef BOOST_INTERPROCESS_DETAIL_WINAPI_WRAPPER_COMMON_HPP
#define BOOST_INTERPROCESS_DETAIL_WINAPI_WRAPPER_COMMON_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/win32_api.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/errors.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <limits>

namespace boost {
namespace interprocess {
namespace ipcdetail {

inline bool do_winapi_wait(void *handle, unsigned long dwMilliseconds)
{
unsigned long ret = winapi::wait_for_single_object(handle, dwMilliseconds);
if(ret == winapi::wait_object_0){
return true;
}
else if(ret == winapi::wait_timeout){
return false;
}
else if(ret == winapi::wait_abandoned){ 
winapi::release_mutex(handle);
throw interprocess_exception(owner_dead_error);
}
else{
error_info err = system_error_code();
throw interprocess_exception(err);
}
}

inline bool winapi_wrapper_timed_wait_for_single_object(void *handle, const boost::posix_time::ptime &abs_time)
{
unsigned long time = 0u;
if (abs_time.is_pos_infinity()){
time = winapi::infinite_time;
}
else {
const boost::posix_time::ptime cur_time = microsec_clock::universal_time();
if(abs_time > cur_time){
time = (abs_time - cur_time).total_milliseconds();
}
}
return do_winapi_wait(handle, time);
}

inline void winapi_wrapper_wait_for_single_object(void *handle)
{
(void)do_winapi_wait(handle, winapi::infinite_time);
}

inline bool winapi_wrapper_try_wait_for_single_object(void *handle)
{
return do_winapi_wait(handle, 0u);
}

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
