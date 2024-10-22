
#ifndef BOOST_INTERPROCESS_SYNC_DETAIL_COMMON_ALGORITHMS_HPP
#define BOOST_INTERPROCESS_SYNC_DETAIL_COMMON_ALGORITHMS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <boost/interprocess/sync/spin/wait.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {

template<class MutexType>
bool try_based_timed_lock(MutexType &m, const boost::posix_time::ptime &abs_time)
{
if(abs_time.is_pos_infinity()){
m.lock();
return true;
}
else if(m.try_lock()){
return true;
}
else{
spin_wait swait;
while(microsec_clock::universal_time() < abs_time){
if(m.try_lock()){
return true;
}
swait.yield();
}
return false;
}
}

template<class MutexType>
void try_based_lock(MutexType &m)
{
if(!m.try_lock()){
spin_wait swait;
do{
if(m.try_lock()){
break;
}
else{
swait.yield();
}
}
while(1);
}
}

template<class MutexType>
void timeout_when_locking_aware_lock(MutexType &m)
{
#ifdef BOOST_INTERPROCESS_ENABLE_TIMEOUT_WHEN_LOCKING
boost::posix_time::ptime wait_time
= microsec_clock::universal_time()
+ boost::posix_time::milliseconds(BOOST_INTERPROCESS_TIMEOUT_WHEN_LOCKING_DURATION_MS);
if (!m.timed_lock(wait_time))
{
throw interprocess_exception(timeout_when_locking_error
, "Interprocess mutex timeout when locking. Possible deadlock: "
"owner died without unlocking?");
}
#else
m.lock();
#endif
}

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
