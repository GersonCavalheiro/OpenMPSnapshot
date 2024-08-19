
#ifndef BOOST_INTERPROCESS_RECURSIVE_MUTEX_HPP
#define BOOST_INTERPROCESS_RECURSIVE_MUTEX_HPP

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/detail/common_algorithms.hpp>
#include <boost/assert.hpp>

#if !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && \
(defined(BOOST_INTERPROCESS_POSIX_PROCESS_SHARED) && defined (BOOST_INTERPROCESS_POSIX_RECURSIVE_MUTEXES))
#include <boost/interprocess/sync/posix/recursive_mutex.hpp>
#define BOOST_INTERPROCESS_USE_POSIX
#elif !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && defined (BOOST_INTERPROCESS_WINDOWS)
#include <boost/interprocess/sync/windows/recursive_mutex.hpp>
#define BOOST_INTERPROCESS_USE_WINDOWS
#elif !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
#include <boost/interprocess/sync/spin/recursive_mutex.hpp>
#define BOOST_INTERPROCESS_USE_GENERIC_EMULATION
#endif

#if defined (BOOST_INTERPROCESS_USE_GENERIC_EMULATION)
namespace boost {
namespace interprocess {
namespace ipcdetail{
namespace robust_emulation_helpers {

template<class T>
class mutex_traits;

}}}}

#endif

#endif   


namespace boost {
namespace interprocess {

class interprocess_recursive_mutex
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
interprocess_recursive_mutex(const interprocess_recursive_mutex &);
interprocess_recursive_mutex &operator=(const interprocess_recursive_mutex &);
#endif   
public:
interprocess_recursive_mutex();

~interprocess_recursive_mutex();

void lock();

bool try_lock();

bool timed_lock(const boost::posix_time::ptime &abs_time);

void unlock();
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:

#if defined (BOOST_INTERPROCESS_USE_GENERIC_EMULATION)
#undef BOOST_INTERPROCESS_USE_GENERIC_EMULATION
void take_ownership(){ mutex.take_ownership(); }
friend class ipcdetail::robust_emulation_helpers::mutex_traits<interprocess_recursive_mutex>;
ipcdetail::spin_recursive_mutex mutex;
#elif defined(BOOST_INTERPROCESS_USE_POSIX)
#undef BOOST_INTERPROCESS_USE_POSIX
ipcdetail::posix_recursive_mutex mutex;
#elif defined(BOOST_INTERPROCESS_USE_WINDOWS)
#undef BOOST_INTERPROCESS_USE_WINDOWS
ipcdetail::windows_recursive_mutex mutex;
#else
#error "Unknown platform for interprocess_mutex"
#endif
#endif   
};

}  
}  

namespace boost {
namespace interprocess {

inline interprocess_recursive_mutex::interprocess_recursive_mutex(){}

inline interprocess_recursive_mutex::~interprocess_recursive_mutex(){}

inline void interprocess_recursive_mutex::lock()
{  ipcdetail::timeout_when_locking_aware_lock(mutex);  }

inline bool interprocess_recursive_mutex::try_lock()
{ return mutex.try_lock(); }

inline bool interprocess_recursive_mutex::timed_lock(const boost::posix_time::ptime &abs_time)
{ return mutex.timed_lock(abs_time); }

inline void interprocess_recursive_mutex::unlock()
{ mutex.unlock(); }

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
