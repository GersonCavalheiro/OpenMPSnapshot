
#ifndef BOOST_INTERPROCESS_SEMAPHORE_HPP
#define BOOST_INTERPROCESS_SEMAPHORE_HPP

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

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/detail/posix_time_types_wrk.hpp>
#include <boost/interprocess/sync/detail/locks.hpp>
#include <boost/interprocess/sync/detail/common_algorithms.hpp>

#if !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && \
(defined(BOOST_INTERPROCESS_POSIX_PROCESS_SHARED) && defined(BOOST_INTERPROCESS_POSIX_UNNAMED_SEMAPHORES))
#include <boost/interprocess/sync/posix/semaphore.hpp>
#define BOOST_INTERPROCESS_USE_POSIX
#elif !defined(BOOST_INTERPROCESS_FORCE_GENERIC_EMULATION) && defined (BOOST_INTERPROCESS_WINDOWS)
#include <boost/interprocess/sync/windows/semaphore.hpp>
#define BOOST_INTERPROCESS_USE_WINDOWS
#elif !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
#include <boost/interprocess/sync/spin/semaphore.hpp>
#define BOOST_INTERPROCESS_USE_GENERIC_EMULATION
#endif

#endif   


namespace boost {
namespace interprocess {

class interprocess_semaphore
{
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
interprocess_semaphore(const interprocess_semaphore &);
interprocess_semaphore &operator=(const interprocess_semaphore &);
#endif   
public:
interprocess_semaphore(unsigned int initialCount);

~interprocess_semaphore();

void post();

void wait();

bool try_wait();

bool timed_wait(const boost::posix_time::ptime &abs_time);

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
#if defined(BOOST_INTERPROCESS_USE_GENERIC_EMULATION)
#undef BOOST_INTERPROCESS_USE_GENERIC_EMULATION
typedef ipcdetail::spin_semaphore internal_sem_t;
#elif defined(BOOST_INTERPROCESS_USE_WINDOWS)
#undef BOOST_INTERPROCESS_USE_WINDOWS
typedef ipcdetail::windows_semaphore internal_sem_t;
#else
#undef BOOST_INTERPROCESS_USE_POSIX
typedef ipcdetail::posix_semaphore internal_sem_t;
#endif   
internal_sem_t m_sem;
#endif   
};

}  
}  

namespace boost {
namespace interprocess {

inline interprocess_semaphore::interprocess_semaphore(unsigned int initialCount)
: m_sem(initialCount)
{}

inline interprocess_semaphore::~interprocess_semaphore(){}

inline void interprocess_semaphore::wait()
{
ipcdetail::lock_to_wait<internal_sem_t> ltw(m_sem);
timeout_when_locking_aware_lock(ltw);
}

inline bool interprocess_semaphore::try_wait()
{ return m_sem.try_wait(); }

inline bool interprocess_semaphore::timed_wait(const boost::posix_time::ptime &abs_time)
{ return m_sem.timed_wait(abs_time); }

inline void interprocess_semaphore::post()
{ m_sem.post(); }

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   
