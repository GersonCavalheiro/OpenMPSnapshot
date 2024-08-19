
#ifndef BOOST_INTERPROCESS_MUTEX_HPP
#define BOOST_INTERPROCESS_MUTEX_HPP

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/assert.hpp>
#include <boost/interprocess/sync/detail/common_algorithms.hpp>


#if !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && defined (BOOST_INTERPROCESS_POSIX_PROCESS_SHARED)
#include <boost/interprocess/sync/posix/mutex.hpp>
#define BOOST_INTERPROCESS_USE_POSIX
#elif !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && defined (BOOST_INTERPROCESS_WINDOWS)
#include <boost/interprocess/sync/windows/mutex.hpp>
#define BOOST_INTERPROCESS_USE_WINDOWS
#elif !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
#include <boost/interprocess/sync/spin/mutex.hpp>
#define BOOST_INTERPROCESS_USE_GENERIC_EMULATION

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

class interprocess_condition;

class interprocess_mutex
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
interprocess_mutex(const interprocess_mutex &);
interprocess_mutex &operator=(const interprocess_mutex &);
friend class interprocess_condition;

public:
#if defined(BOOST_INTERPROCESS_USE_GENERIC_EMULATION)
#undef BOOST_INTERPROCESS_USE_GENERIC_EMULATION
typedef ipcdetail::spin_mutex internal_mutex_type;
private:
friend class ipcdetail::robust_emulation_helpers::mutex_traits<interprocess_mutex>;
void take_ownership(){ m_mutex.take_ownership(); }
public:
#elif defined(BOOST_INTERPROCESS_USE_POSIX)
#undef BOOST_INTERPROCESS_USE_POSIX
typedef ipcdetail::posix_mutex internal_mutex_type;
#elif defined(BOOST_INTERPROCESS_USE_WINDOWS)
#undef BOOST_INTERPROCESS_USE_WINDOWS
typedef ipcdetail::windows_mutex internal_mutex_type;
#else
#error "Unknown platform for interprocess_mutex"
#endif

#endif   
public:

interprocess_mutex();

~interprocess_mutex();

void lock();

bool try_lock();

bool timed_lock(const boost::posix_time::ptime &abs_time);

void unlock();

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
internal_mutex_type &internal_mutex()
{  return m_mutex;   }

const internal_mutex_type &internal_mutex() const
{  return m_mutex;   }

private:
internal_mutex_type m_mutex;
#endif   
};

}  
}  


namespace boost {
namespace interprocess {

inline interprocess_mutex::interprocess_mutex(){}

inline interprocess_mutex::~interprocess_mutex(){}

inline void interprocess_mutex::lock()
{  ipcdetail::timeout_when_locking_aware_lock(m_mutex);  }

inline bool interprocess_mutex::try_lock()
{ return m_mutex.try_lock(); }

inline bool interprocess_mutex::timed_lock(const boost::posix_time::ptime &abs_time)
{ return m_mutex.timed_lock(abs_time); }

inline void interprocess_mutex::unlock()
{ m_mutex.unlock(); }

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
